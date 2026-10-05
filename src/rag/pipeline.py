"""End-to-end RAG pipeline: preprocess -> (intent) -> retrieve -> context -> LLM -> cited answer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from src.llm.base import BaseLLM
from src.nlp.preprocessing import preprocess_query
from src.rag.context import build_context, parse_citations
from src.rag.prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, build_user_prompt
from src.retrieval.retriever import Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Intents that usually need broader evidence get a larger default top_k (only when the caller
# did not set top_k explicitly).
INTENT_TOP_K_MULTIPLIER = {"summary": 2, "comparison": 2}
MAX_AUTO_TOP_K = 20


class IntentClassifierLike(Protocol):
    def predict(self, texts: list[str]) -> list[Any]: ...


@dataclass
class RAGSource:
    document: str
    chunk_id: str
    score: float
    vector_score: float
    page: Optional[int]
    section: Optional[str]
    source_path: str
    text: str
    citation_index: int
    cited: bool


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[RAGSource] = field(default_factory=list)
    model: str = ""
    retrieved_chunks: int = 0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    llm_called: bool = False
    context_words: int = 0
    top_k_used: int = 0
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None


class RAGPipeline:
    """Retrieval-augmented generation. The LLM is never called without retrieved context."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        *,
        default_top_k: int = 5,
        max_context_words: int = 1500,
        intent_predictor: Optional[IntentClassifierLike] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.default_top_k = default_top_k
        self.max_context_words = max_context_words
        self.intent_predictor = intent_predictor

    def _predict_intent(self, query: str) -> tuple[Optional[str], Optional[float]]:
        if self.intent_predictor is None:
            return None, None
        try:
            prediction = self.intent_predictor.predict([query])[0]
            return prediction.label, float(prediction.confidence)
        except Exception as exc:  # intent is an optimisation; never fail the request because of it
            logger.warning("intent_prediction_failed", extra={"error": str(exc)})
            return None, None

    def _effective_top_k(self, requested: Optional[int], intent: Optional[str]) -> int:
        if requested is not None:
            return requested
        factor = INTENT_TOP_K_MULTIPLIER.get(intent or "", 1)
        return min(self.default_top_k * factor, MAX_AUTO_TOP_K)

    def answer(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """Answer ``question`` using retrieved context only."""
        started = time.perf_counter()
        query = preprocess_query(question)
        intent, confidence = self._predict_intent(query)
        k = self._effective_top_k(top_k, intent)

        retrieval = self.retriever.retrieve(query, top_k=k)
        result = RAGResult(
            question=query, answer=NO_CONTEXT_ANSWER, model=self.llm.model_name,
            retrieval_ms=retrieval.total_ms, top_k_used=k, intent=intent, intent_confidence=confidence,
        )
        if not retrieval.chunks:
            result.total_ms = (time.perf_counter() - started) * 1000
            return result

        context = build_context(retrieval.chunks, self.max_context_words)
        prompt = build_user_prompt(query, context.text, intent)
        response = self.llm.generate(SYSTEM_PROMPT, prompt)
        cited = set(parse_citations(response.text))

        result.answer = response.text or NO_CONTEXT_ANSWER
        result.model = response.model or self.llm.model_name
        result.llm_called = True
        result.llm_ms = response.latency_ms
        result.context_words = context.word_count
        result.retrieved_chunks = len(context.used)
        result.sources = [
            RAGSource(
                document=rc.chunk.document, chunk_id=rc.chunk.chunk_id, score=round(rc.score, 6),
                vector_score=round(rc.vector_score, 6), page=rc.chunk.page, section=rc.chunk.section,
                source_path=rc.chunk.source_path, text=rc.chunk.text, citation_index=i, cited=i in cited,
            )
            for i, rc in enumerate(context.used, start=1)
        ]
        result.total_ms = (time.perf_counter() - started) * 1000
        return result
