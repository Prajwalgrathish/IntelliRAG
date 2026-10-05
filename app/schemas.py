"""Pydantic request/response models."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    index_loaded: bool
    indexed_chunks: int


class IngestedDocument(BaseModel):
    document: str
    doc_id: str
    file_type: str
    segments: int
    chunks: int
    words: int


class IngestFailure(BaseModel):
    path: str
    code: str
    reason: str


class IngestResponse(BaseModel):
    ingested: list[IngestedDocument]
    failed: list[IngestFailure]
    total_chunks: int


class IndexRequest(BaseModel):
    rebuild: bool = Field(False, description="Clear the index and re-embed everything.")
    document_ids: Optional[list[str]] = Field(None, description="Restrict indexing to these processed doc_ids.")

    @model_validator(mode="after")
    def _no_partial_rebuild(self) -> "IndexRequest":
        if self.rebuild and self.document_ids:
            raise ValueError("rebuild=true cannot be combined with document_ids")
        return self


class IndexResponse(BaseModel):
    new_chunks: int
    skipped_existing: int
    total_chunks: int
    documents: int
    embedding_ms: float
    embedding_model: str
    saved_to: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Defaults to TOP_K from configuration.")

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return value


class SourceOut(BaseModel):
    document: str
    chunk_id: str
    score: float
    vector_score: float
    page: Optional[int] = None
    section: Optional[str] = None
    source_path: str
    text: str
    citation_index: int
    cited: bool


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    model: str
    retrieved_chunks: int
    latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    llm_called: bool
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None


class DocumentInfo(BaseModel):
    doc_id: str
    document: str
    file_type: Optional[str] = None
    chunks: int
    words: Optional[int] = None
    indexed_chunks: int
    ingested_at: Optional[str] = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_indexed_chunks: int


class ModelInfoResponse(BaseModel):
    app_version: str
    embedding: dict[str, Any]
    classification: dict[str, Any]
    llm: dict[str, Any]
    retrieval: dict[str, Any]
    vector_index: dict[str, Any]
