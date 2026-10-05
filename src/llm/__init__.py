"""LLM backends and factory."""
from __future__ import annotations

from typing import Optional

from src.llm.api import APILLM
from src.llm.base import BaseLLM, LLMError, LLMResponse
from src.llm.local import LocalHFLLM


def create_llm(
    provider: str,
    model_name: str,
    *,
    device: str = "auto",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    api_base_url: str = "",
    api_key: Optional[str] = None,
    api_timeout_s: float = 60.0,
) -> BaseLLM:
    """Create an LLM backend by provider name (``local`` or ``api``)."""
    provider = provider.lower()
    if provider == "local":
        return LocalHFLLM(model_name, device=device, max_new_tokens=max_new_tokens, temperature=temperature)
    if provider == "api":
        return APILLM(
            model_name, base_url=api_base_url, api_key=api_key, timeout_s=api_timeout_s,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}' (expected 'local' or 'api').")


__all__ = ["APILLM", "BaseLLM", "LLMError", "LLMResponse", "LocalHFLLM", "create_llm"]
