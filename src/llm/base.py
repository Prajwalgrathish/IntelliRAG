"""LLM abstraction used by the RAG pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


class LLMError(Exception):
    """Raised when text generation fails (provider error, bad config, timeout)."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class BaseLLM(ABC):
    """Provider-independent text generation interface."""

    provider: str = "base"
    model_name: str = ""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a completion for the given system and user prompts."""

    def info(self) -> dict[str, Any]:
        return {"provider": self.provider, "model_name": self.model_name}
