"""OpenAI-compatible chat-completions backend. Credentials come from the environment only."""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from src.llm.base import BaseLLM, LLMError, LLMResponse


class APILLM(BaseLLM):
    provider = "api"

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str],
        timeout_s: float = 60.0,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if not api_key:
            raise LLMError("LLM_PROVIDER=api requires LLM_API_KEY to be set in the environment.")
        if not base_url:
            raise LLMError("LLM_PROVIDER=api requires LLM_API_BASE_URL.")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client = client

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        started = time.perf_counter()
        try:
            if self._client is not None:
                response = self._client.post(url, json=payload, headers=headers, timeout=self.timeout_s)
            else:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM API request failed: {exc.__class__.__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"LLM API returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM API response format: {exc}") from exc
        return LLMResponse(
            text=text.strip(), model=data.get("model", self.model_name), provider=self.provider,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
        )

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "model_name": self.model_name, "base_url": self.base_url,
            "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
        }
