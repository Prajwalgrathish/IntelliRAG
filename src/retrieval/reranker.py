"""Lightweight lexical reranker.

Combines the dense cosine score with the fraction of the query's content tokens that appear
in the chunk. It is a cheap hybrid signal that helps with exact terms (IDs, names, numbers)
which dense embeddings can blur. It is not a cross-encoder.
"""
from __future__ import annotations

from src.nlp.preprocessing import content_tokens
from src.retrieval.types import RetrievedChunk


class LexicalReranker:
    def __init__(self, alpha: float = 0.8) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return candidates sorted by ``alpha*vector + (1-alpha)*overlap`` (descending)."""
        query_tokens = set(content_tokens(query))
        rescored: list[RetrievedChunk] = []
        for cand in candidates:
            overlap = 0.0
            if query_tokens:
                overlap = len(query_tokens & set(content_tokens(cand.chunk.text))) / len(query_tokens)
            rescored.append(
                RetrievedChunk(
                    chunk=cand.chunk,
                    vector_score=cand.vector_score,
                    score=self.alpha * cand.vector_score + (1 - self.alpha) * overlap,
                    rank=cand.rank,
                    lexical_overlap=overlap,
                )
            )
        return sorted(rescored, key=lambda r: r.score, reverse=True)
