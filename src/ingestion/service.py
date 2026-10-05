"""Ingestion service: load -> clean -> chunk -> persist processed chunks (JSONL + metadata)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from src.ingestion import SUPPORTED_EXTENSIONS, load_document
from src.ingestion.base import DocumentLoadError, Segment
from src.nlp.chunking import Chunk, SentenceChunker
from src.nlp.cleaning import clean_text
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestResult:
    document: str
    doc_id: str
    source_path: str
    file_type: str
    segments: int
    chunks: int
    words: int
    processed_path: str


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "document"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_segments(segments: Iterable[Segment]) -> list[Segment]:
    """Clean each segment's text and drop segments that become empty."""
    cleaned = [Segment(text=clean_text(s.text), page=s.page, section=s.section) for s in segments]
    return [s for s in cleaned if s.text]


def ingest_file(path: Union[str, Path], processed_dir: Union[str, Path], chunker: SentenceChunker) -> IngestResult:
    """Ingest one document and write ``<doc_id>.jsonl`` + ``<doc_id>.meta.json``.

    Raises:
        DocumentLoadError: when the file cannot be loaded, is empty, or yields no chunks.
    """
    path = Path(path)
    processed = Path(processed_dir)
    document = load_document(path)
    sha = _sha256(path)
    doc_id = f"{slugify(path.stem)}-{sha[:8]}"

    segments = preprocess_segments(document.segments)
    if not segments:
        raise DocumentLoadError(path, "empty_after_cleaning", "No text remained after cleaning.")
    chunks = chunker.chunk_document(
        doc_id=doc_id, document_name=document.name, source_path=str(path), segments=segments
    )
    if not chunks:
        raise DocumentLoadError(path, "no_chunks", "Document produced no chunks.")

    processed.mkdir(parents=True, exist_ok=True)
    _remove_stale_versions(processed, document.name, keep_doc_id=doc_id)
    jsonl_path = processed / f"{doc_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    words = sum(c.word_count for c in chunks)
    meta = {
        "doc_id": doc_id,
        "document": document.name,
        "source_path": str(path),
        "file_type": document.file_type,
        "content_sha256": sha,
        "segments": len(segments),
        "chunks": len(chunks),
        "words": words,
        "loader_metadata": document.metadata,
        "chunk_size": chunker.chunk_size,
        "chunk_overlap": chunker.chunk_overlap,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    (processed / f"{doc_id}.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("document_ingested", extra={"document": document.name, "doc_id": doc_id, "chunks": len(chunks)})
    return IngestResult(
        document=document.name, doc_id=doc_id, source_path=str(path), file_type=document.file_type,
        segments=len(segments), chunks=len(chunks), words=words, processed_path=str(jsonl_path),
    )


def _remove_stale_versions(processed: Path, document_name: str, keep_doc_id: str) -> None:
    """Delete processed outputs of older versions of the same file name."""
    for meta_file in processed.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("document") == document_name and meta.get("doc_id") != keep_doc_id:
            doc_id = meta["doc_id"]
            (processed / f"{doc_id}.jsonl").unlink(missing_ok=True)
            meta_file.unlink(missing_ok=True)


def ingest_directory(
    input_dir: Union[str, Path], processed_dir: Union[str, Path], chunker: SentenceChunker, recursive: bool = True
) -> tuple[list[IngestResult], list[dict[str, Any]]]:
    """Ingest every file in a directory. Failures are returned, never silently dropped."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise DocumentLoadError(input_dir, "directory_not_found", "Input directory does not exist.")
    pattern = "**/*" if recursive else "*"
    results: list[IngestResult] = []
    errors: list[dict[str, Any]] = []
    for path in sorted(p for p in input_dir.glob(pattern) if p.is_file() and not p.name.startswith(".")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(
                DocumentLoadError(path, "unsupported_type", f"Skipped unsupported file type '{path.suffix}'.").to_dict()
            )
            continue
        try:
            results.append(ingest_file(path, processed_dir, chunker))
        except DocumentLoadError as exc:
            errors.append(exc.to_dict())
    return results, errors


def load_processed_chunks(processed_dir: Union[str, Path], doc_ids: Optional[Iterable[str]] = None) -> list[Chunk]:
    """Read processed chunk files, optionally restricted to ``doc_ids``."""
    processed = Path(processed_dir)
    wanted = set(doc_ids) if doc_ids else None
    chunks: list[Chunk] = []
    if not processed.is_dir():
        return chunks
    for file in sorted(processed.glob("*.jsonl")):
        if wanted is not None and file.stem not in wanted:
            continue
        with file.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(Chunk.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"Malformed chunk record in {file.name}:{line_no}: {exc}") from exc
    return chunks


def list_processed_documents(processed_dir: Union[str, Path]) -> list[dict[str, Any]]:
    """Return metadata for every processed document."""
    processed = Path(processed_dir)
    docs: list[dict[str, Any]] = []
    if not processed.is_dir():
        return docs
    for meta_file in sorted(processed.glob("*.meta.json")):
        try:
            docs.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable_meta_file", extra={"file": meta_file.name})
    return docs
