"""Test doubles: a deterministic hashing encoder (no model download) and a fake LLM."""
from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.config import Settings
from src.llm.base import BaseLLM, LLMResponse


class HashingEncoder:
    """Bag-of-words feature-hashing encoder. Texts that share content words get higher cosine."""

    model_name = "hashing-test-encoder"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in ENGLISH_STOP_WORDS:
                continue
            vec[zlib.crc32(token.encode()) % self.dimension] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode_documents(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed(t) for t in texts]).astype(np.float32)

    encode_queries = encode_documents


class FakeLLM(BaseLLM):
    provider = "fake"
    model_name = "fake-llm"

    def __init__(self, reply: str = "The documents describe the requested topic. [1]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        return LLMResponse(text=self.reply, model=self.model_name, provider=self.provider, latency_ms=1.5)


def make_settings(root: Path, **overrides) -> Settings:
    """Settings that point every path at a temporary directory."""
    values = dict(
        documents_dir=str(root / "documents"), processed_dir=str(root / "processed"),
        vector_db_path=str(root / "vector_store"), models_dir=str(root / "models"),
        logs_dir=str(root / "logs"), evaluation_dir=str(root / "evaluation"),
        intent_model_path=str(root / "models" / "intent.pt"),
        min_similarity=0.0, rerank_enabled=True, chunk_size=60, chunk_overlap=10,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)
