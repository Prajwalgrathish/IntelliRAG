"""Application settings loaded from environment variables / .env (no secrets in code)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False, protected_namespaces=()
    )

    app_name: str = "IntelliRAG"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Models (all configurable)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    classification_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_pooling: Literal["mean", "cls"] = "mean"
    embedding_max_length: int = Field(256, ge=8)
    embedding_batch_size: int = Field(32, ge=1)
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    device: str = "auto"

    # LLM provider
    llm_provider: Literal["local", "api"] = "local"
    llm_max_new_tokens: int = Field(256, ge=1)
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_api_base_url: str = "https://api.openai.com/v1"
    llm_api_key: Optional[SecretStr] = None
    llm_api_timeout_s: float = Field(60.0, gt=0)

    # Storage
    vector_db_path: str = "vector_store"
    documents_dir: str = "data/documents"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    logs_dir: str = "logs"
    evaluation_dir: str = "evaluation"
    intent_model_path: str = "models/intent_classifier.pt"

    # Retrieval / chunking
    top_k: int = Field(5, ge=1, le=50)
    chunk_size: int = Field(200, ge=10)
    chunk_overlap: int = Field(40, ge=0)
    min_similarity: float = Field(0.2, ge=-1.0, le=1.0)
    rerank_enabled: bool = True
    rerank_alpha: float = Field(0.8, ge=0.0, le=1.0)
    rerank_candidate_multiplier: int = Field(3, ge=1, le=20)
    max_context_words: int = Field(1500, ge=50)

    max_upload_mb: int = Field(25, ge=1)

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    def llm_api_key_value(self) -> Optional[str]:
        """Return the API key (or None when unset/empty)."""
        if self.llm_api_key is None:
            return None
        value = self.llm_api_key.get_secret_value().strip()
        return value or None

    def ensure_directories(self) -> None:
        for directory in (
            self.documents_dir, self.processed_dir, self.models_dir, self.logs_dir, self.evaluation_dir, self.vector_db_path,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()
