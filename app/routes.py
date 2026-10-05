"""API endpoints."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile

from app.dependencies import ServiceContainer, get_container
from app.errors import ApiError
from app.schemas import (
    DocumentInfo, DocumentsResponse, HealthResponse, IndexRequest, IndexResponse, IngestedDocument,
    IngestFailure, IngestResponse, ModelInfoResponse, QueryRequest, QueryResponse, SourceOut,
)
from src.ingestion import SUPPORTED_EXTENSIONS
from src.ingestion.base import DocumentLoadError
from src.ingestion.service import ingest_file, list_processed_documents, load_processed_chunks
from src.vectorstore.indexer import index_chunks

router = APIRouter()
_STARTED = time.monotonic()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(c: ServiceContainer = Depends(get_container)) -> HealthResponse:
    store = c.store
    return HealthResponse(
        status="ok", version=c.settings.app_version, uptime_seconds=round(time.monotonic() - _STARTED, 2),
        index_loaded=not store.is_empty, indexed_chunks=len(store),
    )


@router.post("/documents/ingest", response_model=IngestResponse, tags=["documents"])
def ingest_documents(files: list[UploadFile] = File(...), c: ServiceContainer = Depends(get_container)) -> IngestResponse:
    """Upload documents; extract text, clean, chunk and store processed chunks (does not embed)."""
    docs_dir = Path(c.settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = c.settings.max_upload_mb * 1024 * 1024
    ingested: list[IngestedDocument] = []
    failed: list[dict] = []

    for upload in files:
        filename = Path(upload.filename or "").name
        target = docs_dir / filename if filename else None
        previous: Optional[bytes] = None
        wrote = False
        try:
            if not filename:
                raise DocumentLoadError("", "invalid_filename", "Uploaded file has no name.")
            if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise DocumentLoadError(
                    filename, "unsupported_type",
                    f"Unsupported file type '{Path(filename).suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                )
            data = upload.file.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise DocumentLoadError(filename, "file_too_large", f"File exceeds {c.settings.max_upload_mb} MB limit.")
            assert target is not None
            previous = target.read_bytes() if target.exists() else None
            target.write_bytes(data)
            wrote = True
            result = ingest_file(target, c.settings.processed_dir, c.chunker)
            ingested.append(
                IngestedDocument(
                    document=result.document, doc_id=result.doc_id, file_type=result.file_type,
                    segments=result.segments, chunks=result.chunks, words=result.words,
                )
            )
        except DocumentLoadError as exc:
            failed.append(exc.to_dict())
            if wrote and target is not None:  # roll back only what this request wrote
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(previous)

    total_chunks = sum(d.chunks for d in ingested)
    c.metrics.record_ingestion(len(ingested), total_chunks, failures=len(failed))
    c.events.log("ingest", ingested=len(ingested), failed=len(failed), chunks=total_chunks)
    if not ingested:
        raise ApiError(422, "ingestion_failed", "No documents could be ingested.", details={"failed": failed})
    return IngestResponse(ingested=ingested, failed=[IngestFailure(**f) for f in failed], total_chunks=total_chunks)


@router.post("/documents/index", response_model=IndexResponse, tags=["documents"])
def index_documents(req: IndexRequest, c: ServiceContainer = Depends(get_container)) -> IndexResponse:
    """Embed processed chunks and add them to the FAISS index, then persist it."""
    chunks = load_processed_chunks(c.settings.processed_dir, req.document_ids)
    if not chunks:
        raise ApiError(
            409, "nothing_to_index", "No processed chunks found. Ingest documents first (POST /documents/ingest)."
        )
    result = index_chunks(c.encoder, c.store, chunks, batch_size=c.settings.embedding_batch_size, rebuild=req.rebuild)
    if len(c.store) == 0:
        raise ApiError(409, "nothing_to_index", "The index is still empty after indexing.")
    c.persist_store()
    c.metrics.record_indexing(result.new_chunks)
    c.events.log("index", new_chunks=result.new_chunks, total_chunks=result.total_chunks, embedding_ms=result.embedding_ms)
    return IndexResponse(
        new_chunks=result.new_chunks, skipped_existing=result.skipped_existing, total_chunks=result.total_chunks,
        documents=result.documents, embedding_ms=result.embedding_ms, embedding_model=c.encoder.model_name,
        saved_to=str(c.vector_path),
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest, c: ServiceContainer = Depends(get_container)) -> QueryResponse:
    """Answer a question using retrieved document context only."""
    if c.store.is_empty:
        raise ApiError(
            409, "index_empty",
            "The vector index is empty. Ingest documents (POST /documents/ingest) and build the index (POST /documents/index) first.",
        )
    try:
        result = c.pipeline.answer(req.question, top_k=req.top_k)
    except ValueError as exc:
        c.metrics.record_query(success=False)
        raise ApiError(422, "invalid_query", str(exc)) from exc
    except Exception:
        c.metrics.record_query(success=False)
        raise
    c.metrics.record_query(True, retrieval_ms=result.retrieval_ms, llm_ms=result.llm_ms, total_ms=result.total_ms)
    c.events.log(
        "query", question_chars=len(req.question), top_k=result.top_k_used, retrieved=result.retrieved_chunks,
        llm_called=result.llm_called, retrieval_ms=round(result.retrieval_ms, 2), llm_ms=round(result.llm_ms, 2),
        total_ms=round(result.total_ms, 2), model=result.model, intent=result.intent,
    )
    return QueryResponse(
        answer=result.answer,
        sources=[SourceOut(**vars(s)) for s in result.sources],
        model=result.model, retrieved_chunks=result.retrieved_chunks, latency_ms=round(result.total_ms, 2),
        retrieval_latency_ms=round(result.retrieval_ms, 2), llm_latency_ms=round(result.llm_ms, 2),
        llm_called=result.llm_called, intent=result.intent, intent_confidence=result.intent_confidence,
    )


@router.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents(c: ServiceContainer = Depends(get_container)) -> DocumentsResponse:
    """List ingested documents and how many of their chunks are indexed."""
    indexed_by_doc = c.store.document_counts()
    docs = []
    seen = set()
    for meta in list_processed_documents(c.settings.processed_dir):
        seen.add(meta["document"])
        docs.append(
            DocumentInfo(
                doc_id=meta["doc_id"], document=meta["document"], file_type=meta.get("file_type"),
                chunks=meta.get("chunks", 0), words=meta.get("words"),
                indexed_chunks=indexed_by_doc.get(meta["document"], 0), ingested_at=meta.get("ingested_at"),
            )
        )
    return DocumentsResponse(documents=docs, total_documents=len(docs), total_indexed_chunks=len(c.store))


@router.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info(c: ServiceContainer = Depends(get_container)) -> ModelInfoResponse:
    """Report configured/loaded models and index statistics (does not trigger model loading)."""
    return ModelInfoResponse(**c.model_info())


@router.get("/metrics", tags=["system"])
def metrics(c: ServiceContainer = Depends(get_container)) -> dict:
    """Request, latency, ingestion and error metrics for this process."""
    return c.metrics.snapshot()
