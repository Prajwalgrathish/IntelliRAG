"""Shared retrieval data types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.nlp.chunking import Chunk


@dataclass
class RetrievedChunk:
    """A retrieved chunk. ``score`` is the final ranking score; ``vector_score`` is cosine similarity."""

    chunk: Chunk
    vector_score: float
    score: float
    rank: int = 0
    lexical_overlap: Optional[float] = None


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    candidates_considered: int = 0
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
