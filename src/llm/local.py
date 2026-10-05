"""Local Hugging Face causal-LM backend (lazy-loaded on first use)."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from src.llm.base import BaseLLM, LLMError, LLMResponse
from src.utils.device import resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LocalHFLLM(BaseLLM):
    provider = "local"

    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 256, temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.device_preference = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device = resolve_device(self.device_preference)
                logger.info("loading_llm", extra={"model": self.model_name, "device": device})
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device).eval()
            except Exception as exc:
                raise LLMError(f"Failed to load local LLM '{self.model_name}': {exc}") from exc
            self._tokenizer, self._model, self._device = tokenizer, model, device

    def _render_prompt(self, system_prompt: str, user_prompt: str) -> str:
        if getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{system_prompt}\n\n{user_prompt}\n"

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import torch

        self._ensure_loaded()
        started = time.perf_counter()
        try:
            prompt = self._render_prompt(system_prompt, user_prompt)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            }
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                output = self._model.generate(**inputs, **kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output[0][prompt_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as exc:
            raise LLMError(f"Local generation failed: {exc}") from exc
        return LLMResponse(
            text=text, model=self.model_name, provider=self.provider,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=int(prompt_len), completion_tokens=int(len(new_tokens)),
        )

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "model_name": self.model_name, "loaded": self.loaded,
            "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
        }
