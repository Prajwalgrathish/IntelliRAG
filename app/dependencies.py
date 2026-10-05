"""Service container (lazy, injectable) and FastAPI dependency accessors."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import Request

from app.config import Settings
from app.monitoring import MetricsRegistry
from src.llm import BaseLLM, create_llm
from src.nlp.chunking import SentenceChunker
from src.rag.pipeline import RAGPipeline
from src.retrieval import LexicalReranker, Retriever
from src.utils.logging import JsonlEventLogger, get_logger
from src.vectorstore.faiss_store import FaissStore

logger = get_logger(__name__)
_UNSET: Any = object()


class ServiceContainer:
    """Owns long-lived services. Heavy models load lazily on first use.

    Every collaborator can be injected (used by tests and scripts), which keeps the API layer
    free of hard-wired model construction.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        encoder: Any = None,
        llm: Optional[BaseLLM] = None,
        store: Optional[FaissStore] = None,
        intent_predictor: Any = _UNSET,
        metrics: Optional[MetricsRegistry] = None,
        event_logger: Optional[JsonlEventLogger] = None,
    ) -> None:
        self.settings = settings
        self.metrics = metrics or MetricsRegistry()
        self.events = event_logger or JsonlEventLogger(Path(settings.logs_dir) / "requests.jsonl")
        self._encoder = encoder
        self._llm = llm
        self._store = store
        self._intent_predictor = intent_predictor
        self._retriever: Optional[Retriever] = None
        self._pipeline: Optional[RAGPipeline] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths / helpers
    @property
    def vector_path(self) -> Path:
        return Path(self.settings.vector_db_path)

    @property
    def chunker(self) -> SentenceChunker:
        return SentenceChunker(self.settings.chunk_size, self.settings.chunk_overlap)

    # ------------------------------------------------------------------ lazy services
    def _make_encoder(self, model_name: str) -> Any:
        from src.embeddings.transformer_encoder import TransformerEncoder

        s = self.settings
        return TransformerEncoder(
            model_name, device=s.device, batch_size=s.embedding_batch_size, max_length=s.embedding_max_length,
            pooling=s.embedding_pooling, query_prefix=s.embedding_query_prefix, document_prefix=s.embedding_document_prefix,
        )

    @property
    def encoder(self) -> Any:
        with self._lock:
            if self._encoder is None:
                self._encoder = self._make_encoder(self.settings.embedding_model_name)
            return self._encoder

    @property
    def store(self) -> FaissStore:
        with self._lock:
            if self._store is None:
                if FaissStore.exists(self.vector_path):
                    self._store = FaissStore.load(self.vector_path)
                    logger.info("vector_store_loaded", extra={"chunks": len(self._store)})
                else:
                    self._store = FaissStore()
            return self._store

    @property
    def llm(self) -> BaseLLM:
        with self._lock:
            if self._llm is None:
                s = self.settings
                self._llm = create_llm(
                    s.llm_provider, s.llm_model_name, device=s.device, max_new_tokens=s.llm_max_new_tokens,
                    temperature=s.llm_temperature, api_base_url=s.llm_api_base_url,
                    api_key=s.llm_api_key_value(), api_timeout_s=s.llm_api_timeout_s,
                )
            return self._llm

    @property
    def intent_predictor(self) -> Any:
        """Intent classifier if a trained checkpoint exists, else None."""
        with self._lock:
            if self._intent_predictor is _UNSET:
                self._intent_predictor = None
                path = Path(self.settings.intent_model_path)
                if path.is_file():
                    try:
                        from src.neural.model import IntentPredictor
                        from src.utils.device import resolve_device

                        def provider(name: str) -> Any:
                            if name == self.settings.embedding_model_name:
                                return self.encoder
                            return self._make_encoder(name)

                        self._intent_predictor = IntentPredictor.from_checkpoint(
                            path, provider, device=resolve_device(self.settings.device)
                        )
                        logger.info("intent_classifier_loaded", extra={"path": str(path)})
                    except Exception as exc:
                        logger.warning("intent_classifier_load_failed", extra={"error": str(exc)})
            return self._intent_predictor

    @property
    def retriever(self) -> Retriever:
        with self._lock:
            if self._retriever is None:
                s = self.settings
                reranker = LexicalReranker(s.rerank_alpha) if s.rerank_enabled else None
                self._retriever = Retriever(
                    self.encoder, self.store, reranker=reranker, min_similarity=s.min_similarity,
                    candidate_multiplier=s.rerank_candidate_multiplier,
                )
            return self._retriever

    @property
    def pipeline(self) -> RAGPipeline:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = RAGPipeline(
                    self.retriever, self.llm, default_top_k=self.settings.top_k,
                    max_context_words=self.settings.max_context_words, intent_predictor=self.intent_predictor,
                )
            return self._pipeline

    def persist_store(self) -> None:
        self.store.save(self.vector_path)

    # ------------------------------------------------------------------ introspection (never loads models)
    def model_versions(self) -> dict[str, Any]:
        s = self.settings
        return {
            "app_version": s.app_version,
            "embedding_model": s.embedding_model_name,
            "classification_model": s.classification_model_name,
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model_name,
            "intent_model_present": Path(s.intent_model_path).is_file(),
        }

    def model_info(self) -> dict[str, Any]:
        s = self.settings
        store = self.store
        predictor = self._intent_predictor if self._intent_predictor is not _UNSET else None
        dimension = self._encoder.dimension if self._encoder is not None else store.dimension
        llm_info = self._llm.info() if self._llm is not None else {
            "provider": s.llm_provider, "model_name": s.llm_model_name, "loaded": False,
            "max_new_tokens": s.llm_max_new_tokens, "temperature": s.llm_temperature,
        }
        if s.llm_provider == "api":
            llm_info["api_key_configured"] = s.llm_api_key_value() is not None
        return {
            "app_version": s.app_version,
            "embedding": {
                "model_name": s.embedding_model_name, "loaded": self._encoder is not None, "dimension": dimension,
                "pooling": s.embedding_pooling, "max_length": s.embedding_max_length, "normalized": True,
            },
            "classification": {
                "model_name": s.classification_model_name,
                "intent_model_path": s.intent_model_path,
                "intent_model_present": Path(s.intent_model_path).is_file(),
                "intent_classifier_loaded": predictor is not None,
                "labels": getattr(predictor, "labels", None),
            },
            "llm": llm_info,
            "retrieval": {
                "default_top_k": s.top_k, "min_similarity": s.min_similarity, "rerank_enabled": s.rerank_enabled,
                "rerank_alpha": s.rerank_alpha, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
            },
            "vector_index": store.stats(),
        }


def get_container(request: Request) -> ServiceContainer:
    """FastAPI dependency returning the application's service container."""
    return request.app.state.container
