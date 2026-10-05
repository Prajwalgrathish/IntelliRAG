"""Semantic retriever: embed query -> vector search -> threshold -> optional rerank."""
from __future__ import annotations

import time
from typing import Optional

from src.embeddings.transformer_encoder import TextEncoder
from src.nlp.preprocessing import preprocess_query
from src.retrieval.reranker import LexicalReranker
from src.retrieval.types import RetrievalResult, RetrievedChunk
from src.vectorstore.faiss_store import FaissStore


class Retriever:
    def __init__(
        self,
        encoder: TextEncoder,
        store: FaissStore,
        reranker: Optional[LexicalReranker] = None,
        min_similarity: float = 0.0,
        candidate_multiplier: int = 3,
    ) -> None:
        self.encoder = encoder
        self.store = store
        self.reranker = reranker
        self.min_similarity = min_similarity
        self.candidate_multiplier = max(1, candidate_multiplier)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve the ``top_k`` most relevant chunks for ``query``.

        Chunks whose cosine similarity is below ``min_similarity`` are dropped *before*
        reranking, so an off-topic question can legitimately return zero chunks.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        started = time.perf_counter()
        processed = preprocess_query(query)
        result = RetrievalResult(query=processed)
        if self.store.is_empty:
            return result
        self.store.assert_compatible(self.encoder.model_name, self.encoder.dimension)

        t0 = time.perf_counter()
        vector = self.encoder.encode_queries([processed])[0]
        result.embedding_ms = (time.perf_counter() - t0) * 1000

        fetch_k = top_k * self.candidate_multiplier if self.reranker else top_k
        t1 = time.perf_counter()
        hits = self.store.search(vector, fetch_k)
        result.search_ms = (time.perf_counter() - t1) * 1000
        candidates = [
            RetrievedChunk(chunk=c, vector_score=s, score=s) for c, s in hits if s >= self.min_similarity
        ]
        result.candidates_considered = len(candidates)

        if self.reranker and candidates:
            t2 = time.perf_counter()
            candidates = self.reranker.rerank(processed, candidates)
            result.rerank_ms = (time.perf_counter() - t2) * 1000

        result.chunks = candidates[:top_k]
        for rank, item in enumerate(result.chunks, start=1):
            item.rank = rank
        result.total_ms = (time.perf_counter() - started) * 1000
        return result
