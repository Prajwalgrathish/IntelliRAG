"""Pure metric functions for retrieval and generation evaluation."""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

from src.nlp.preprocessing import content_tokens


def recall_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    if total_relevant <= 0:
        raise ValueError("total_relevant must be > 0")
    return min(sum(relevance[:k]), total_relevant) / total_relevant


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Fraction of the k requested slots that hold relevant items (missing slots count as misses)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    return sum(relevance[:k]) / k


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 if at least one relevant item is in the top-k."""
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """1/rank of the first relevant item (0.0 when none)."""
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean of non-None values, or None if there are none."""
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-content-tokens F1 between a prediction and a reference answer."""
    pred, ref = Counter(content_tokens(prediction)), Counter(content_tokens(reference))
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)
