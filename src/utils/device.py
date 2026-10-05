"""Compute-device selection."""
from __future__ import annotations


def resolve_device(preference: str = "auto") -> str:
    """Resolve ``auto`` to cuda / mps / cpu; pass explicit choices through."""
    if preference and preference.lower() != "auto":
        return preference.lower()
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
