"""FAISS-backed vector store with JSONL metadata persistence.

Vectors are L2-normalised and indexed in an ``IndexFlatIP`` so inner product equals cosine
similarity. Exact (flat) search is appropriate for portfolio-scale corpora.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import faiss
import numpy as np

from src.nlp.chunking import Chunk


class VectorStoreError(Exception):
    """Raised for invalid vector-store operations or corrupted persisted state."""


class FaissStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.jsonl"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, dimension: Optional[int] = None, embedding_model_name: Optional[str] = None) -> None:
        self._dimension = dimension
        self.embedding_model_name = embedding_model_name
        self._index: Optional[faiss.Index] = faiss.IndexFlatIP(dimension) if dimension else None
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ properties
    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def __len__(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ mutation
    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, embeddings: np.ndarray, chunks: Sequence[Chunk]) -> int:
        """Add chunk embeddings, skipping chunk ids that are already indexed. Returns #added."""
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Embedding matrix shape {embeddings.shape} does not match {len(chunks)} chunks."
            )
        with self._lock:
            if self._index is None:
                self._dimension = int(embeddings.shape[1])
                self._index = faiss.IndexFlatIP(self._dimension)
            if embeddings.shape[1] != self._dimension:
                raise VectorStoreError(
                    f"Embedding dimension {embeddings.shape[1]} != index dimension {self._dimension}."
                )
            keep = [i for i, c in enumerate(chunks) if c.chunk_id not in self._ids]
            if not keep:
                return 0
            vectors = self._normalise(embeddings[keep])
            self._index.add(vectors)
            for i in keep:
                self._chunks.append(chunks[i])
                self._ids.add(chunks[i].chunk_id)
            return len(keep)

    def clear(self) -> None:
        """Remove all vectors and forget the embedding model/dimension."""
        with self._lock:
            self._index = None
            self._dimension = None
            self.embedding_model_name = None
            self._chunks, self._ids = [], set()

    # ------------------------------------------------------------------ query
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return up to ``k`` (chunk, cosine score) pairs, best first."""
        if k < 1:
            raise ValueError("k must be >= 1")
        with self._lock:
            if self._index is None or not self._chunks:
                return []
            query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
            if query.shape[1] != self._dimension:
                raise VectorStoreError(f"Query dimension {query.shape[1]} != index dimension {self._dimension}.")
            scores, ids = self._index.search(self._normalise(query), min(k, len(self._chunks)))
            return [(self._chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def assert_compatible(self, model_name: str, dimension: int) -> None:
        """Ensure the encoder matches the one used to build the index."""
        if self._dimension is not None and self._dimension != dimension:
            raise VectorStoreError(
                f"Index dimension is {self._dimension} but the encoder produces {dimension}. "
                "Rebuild the index (POST /documents/index with rebuild=true)."
            )
        if self.embedding_model_name and self.embedding_model_name != model_name:
            raise VectorStoreError(
                f"Index was built with '{self.embedding_model_name}' but the active encoder is '{model_name}'. "
                "Rebuild the index (POST /documents/index with rebuild=true)."
            )

    # ------------------------------------------------------------------ introspection
    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    def document_counts(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for chunk in self._chunks:
                counts[chunk.document] = counts.get(chunk.document, 0) + 1
            return counts

    def indexed_ids(self) -> set[str]:
        with self._lock:
            return set(self._ids)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chunks": len(self._chunks),
                "documents": len({c.document for c in self._chunks}),
                "dimension": self._dimension,
                "embedding_model_name": self.embedding_model_name,
                "index_type": "IndexFlatIP",
                "metric": "cosine (inner product on L2-normalised vectors)",
            }

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def exists(directory: Union[str, Path]) -> bool:
        directory = Path(directory)
        return (directory / FaissStore.INDEX_FILE).is_file() and (directory / FaissStore.META_FILE).is_file()

    def save(self, directory: Union[str, Path]) -> None:
        """Atomically persist index, metadata (JSONL) and manifest."""
        with self._lock:
            if self._index is None or not self._chunks:
                raise VectorStoreError("Refusing to save an empty vector store.")
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)

            tmp_index = directory / (self.INDEX_FILE + ".tmp")
            faiss.write_index(self._index, str(tmp_index))
            tmp_meta = directory / (self.META_FILE + ".tmp")
            with tmp_meta.open("w", encoding="utf-8") as handle:
                for chunk in self._chunks:
                    handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            manifest = {
                "embedding_model_name": self.embedding_model_name,
                "dimension": self._dimension,
                "num_chunks": len(self._chunks),
                "index_type": "IndexFlatIP",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp_manifest = directory / (self.MANIFEST_FILE + ".tmp")
            tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            os.replace(tmp_index, directory / self.INDEX_FILE)
            os.replace(tmp_meta, directory / self.META_FILE)
            os.replace(tmp_manifest, directory / self.MANIFEST_FILE)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "FaissStore":
        """Load a persisted store and validate that index and metadata agree."""
        directory = Path(directory)
        if not cls.exists(directory):
            raise VectorStoreError(f"No vector store found in '{directory}'.")
        try:
            index = faiss.read_index(str(directory / cls.INDEX_FILE))
            manifest_path = directory / cls.MANIFEST_FILE
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            chunks = [
                Chunk.from_dict(json.loads(line))
                for line in (directory / cls.META_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception as exc:
            raise VectorStoreError(f"Failed to load vector store from '{directory}': {exc}") from exc
        if index.ntotal != len(chunks):
            raise VectorStoreError(
                f"Corrupted store: index has {index.ntotal} vectors but metadata has {len(chunks)} chunks."
            )
        store = cls(embedding_model_name=manifest.get("embedding_model_name"))
        store._index = index
        store._dimension = int(index.d)
        store._chunks = chunks
        store._ids = {c.chunk_id for c in chunks}
        return store
