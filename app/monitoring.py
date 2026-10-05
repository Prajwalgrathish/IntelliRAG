"""Thread-safe in-process metrics registry (counters, latency stats, recent errors)."""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


class LatencyStat:
    """Running count/sum/min/max plus percentiles over a sliding window."""

    def __init__(self, window: int = 1000) -> None:
        self.count = 0
        self.total = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self._recent: deque[float] = deque(maxlen=window)

    def record(self, value_ms: float) -> None:
        self.count += 1
        self.total += value_ms
        self.min = value_ms if self.min is None else min(self.min, value_ms)
        self.max = value_ms if self.max is None else max(self.max, value_ms)
        self._recent.append(value_ms)

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "avg_ms": None, "min_ms": None, "max_ms": None, "p50_ms": None, "p95_ms": None}
        recent = np.asarray(self._recent)
        return {
            "count": self.count,
            "avg_ms": round(self.total / self.count, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "p50_ms": round(float(np.percentile(recent, 50)), 3),
            "p95_ms": round(float(np.percentile(recent, 95)), 3),
        }


class MetricsRegistry:
    """Records request, query, ingestion, indexing and error metrics.

    A request counts as *failed* when its HTTP status is >= 400 (including client errors).
    """

    COUNTERS = (
        "total_requests", "successful_requests", "failed_requests",
        "query_count", "successful_queries", "failed_queries",
        "documents_ingested", "chunks_ingested", "ingestion_failures",
        "indexing_runs", "chunks_indexed", "errors_total",
    )

    def __init__(self, max_recent_errors: int = 50) -> None:
        self._lock = threading.Lock()
        self._started_monotonic = time.monotonic()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self.counters = {name: 0 for name in self.COUNTERS}
        self.latency = {k: LatencyStat() for k in ("request", "retrieval", "llm", "query_total")}
        self.errors_by_type: dict[str, int] = {}
        self.recent_errors: deque[dict[str, Any]] = deque(maxlen=max_recent_errors)
        self.model_versions: dict[str, Any] = {}

    def record_request(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.counters["total_requests"] += 1
            self.counters["successful_requests" if status_code < 400 else "failed_requests"] += 1
            self.latency["request"].record(latency_ms)

    def record_query(
        self, success: bool, retrieval_ms: Optional[float] = None, llm_ms: Optional[float] = None, total_ms: Optional[float] = None
    ) -> None:
        with self._lock:
            self.counters["query_count"] += 1
            self.counters["successful_queries" if success else "failed_queries"] += 1
            if retrieval_ms is not None:
                self.latency["retrieval"].record(retrieval_ms)
            if llm_ms:  # only when the LLM was actually called
                self.latency["llm"].record(llm_ms)
            if total_ms is not None:
                self.latency["query_total"].record(total_ms)

    def record_ingestion(self, documents: int, chunks: int, failures: int = 0) -> None:
        with self._lock:
            self.counters["documents_ingested"] += documents
            self.counters["chunks_ingested"] += chunks
            self.counters["ingestion_failures"] += failures

    def record_indexing(self, chunks: int) -> None:
        with self._lock:
            self.counters["indexing_runs"] += 1
            self.counters["chunks_indexed"] += chunks

    def record_error(self, error_type: str, message: str, path: Optional[str] = None) -> None:
        with self._lock:
            self.counters["errors_total"] += 1
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
            self.recent_errors.append(
                {"timestamp": datetime.now(timezone.utc).isoformat(), "type": error_type, "message": message[:500], "path": path}
            )

    def set_model_versions(self, versions: dict[str, Any]) -> None:
        with self._lock:
            self.model_versions = dict(versions)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self._started_at,
                "uptime_seconds": round(time.monotonic() - self._started_monotonic, 2),
                "counters": dict(self.counters),
                "latency": {name: stat.summary() for name, stat in self.latency.items()},
                "errors_by_type": dict(self.errors_by_type),
                "recent_errors": list(self.recent_errors),
                "model_versions": dict(self.model_versions),
            }
