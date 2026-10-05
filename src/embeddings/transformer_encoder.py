"""Hugging Face transformer text encoder (mean/CLS pooling + L2 normalisation)."""
from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.device import resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class TextEncoder(Protocol):
    """Interface used by retrieval, indexing, classification and evaluation code."""

    model_name: str
    dimension: int

    def encode_queries(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray: ...

    def encode_documents(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray: ...


def pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool token embeddings into one vector per text.

    ``mean`` averages over non-padding tokens; ``cls`` takes the first token.
    """
    if mode == "cls":
        return last_hidden_state[:, 0]
    if mode != "mean":
        raise ValueError(f"Unknown pooling mode '{mode}' (expected 'mean' or 'cls').")
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TransformerEncoder:
    """Encode text with any Hugging Face encoder model (``AutoModel``)."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 256,
        pooling: str = "mean",
        normalize: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

        logger.info("loading_encoder", extra={"model": model_name, "device": self.device})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dimension = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def _encode(self, texts: Sequence[str], batch_size: Optional[int]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        size = batch_size or self.batch_size
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            vectors = pool_embeddings(hidden, encoded["attention_mask"], self.pooling)
            if self.normalize:
                vectors = F.normalize(vectors, p=2, dim=1)
            outputs.append(vectors.float().cpu().numpy())
        return np.vstack(outputs).astype(np.float32)

    def encode_queries(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode user queries (applies ``query_prefix``)."""
        return self._encode([self.query_prefix + t for t in texts], batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode document chunks (applies ``document_prefix``)."""
        return self._encode([self.document_prefix + t for t in texts], batch_size)
