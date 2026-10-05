"""Embed chunks and add them to a vector store."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

from src.embeddings.transformer_encoder import TextEncoder
from src.nlp.chunking import Chunk
from src.vectorstore.faiss_store import FaissStore


@dataclass
class IndexResult:
    new_chunks: int
    skipped_existing: int
    total_chunks: int
    documents: int
    embedding_ms: float


def index_chunks(
    encoder: TextEncoder,
    store: FaissStore,
    chunks: Sequence[Chunk],
    batch_size: Optional[int] = None,
    rebuild: bool = False,
) -> IndexResult:
    """Embed ``chunks`` and add the new ones to ``store`` (optionally clearing it first)."""
    if rebuild:
        store.clear()
    store.assert_compatible(encoder.model_name, encoder.dimension)

    existing = store.indexed_ids()
    fresh = [c for c in chunks if c.chunk_id not in existing]
    start = time.perf_counter()
    added = 0
    if fresh:
        vectors = encoder.encode_documents([c.text for c in fresh], batch_size=batch_size)
        added = store.add(vectors, fresh)
        if store.embedding_model_name is None:
            store.embedding_model_name = encoder.model_name
    elapsed = (time.perf_counter() - start) * 1000
    return IndexResult(
        new_chunks=added,
        skipped_existing=len(chunks) - len(fresh),
        total_chunks=len(store),
        documents=len(store.document_counts()),
        embedding_ms=round(elapsed, 2),
    )
