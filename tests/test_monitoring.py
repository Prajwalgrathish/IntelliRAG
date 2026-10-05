import json
import threading

import pytest

from app.monitoring import LatencyStat, MetricsRegistry
from src.utils.logging import JsonFormatter, JsonlEventLogger
import logging


def test_latency_stat_summary():
    stat = LatencyStat()
    assert stat.summary()["avg_ms"] is None
    for value in (10.0, 20.0, 30.0, 40.0):
        stat.record(value)
    summary = stat.summary()
    assert summary["count"] == 4 and summary["avg_ms"] == 25.0
    assert summary["min_ms"] == 10.0 and summary["max_ms"] == 40.0
    assert summary["p50_ms"] == 25.0 and 38 < summary["p95_ms"] <= 40


def test_request_counters_split_success_and_failure():
    m = MetricsRegistry()
    m.record_request("/health", "GET", 200, 5.0)
    m.record_request("/query", "POST", 422, 3.0)
    m.record_request("/query", "POST", 500, 7.0)
    snap = m.snapshot()
    assert snap["counters"]["total_requests"] == 3
    assert snap["counters"]["successful_requests"] == 1 and snap["counters"]["failed_requests"] == 2
    assert snap["latency"]["request"]["avg_ms"] == 5.0


def test_query_ingestion_indexing_and_error_tracking():
    m = MetricsRegistry()
    m.record_query(True, retrieval_ms=10, llm_ms=100, total_ms=115)
    m.record_query(True, retrieval_ms=20, llm_ms=0, total_ms=21)  # no LLM call -> not counted in LLM latency
    m.record_query(False)
    m.record_ingestion(documents=2, chunks=30, failures=1)
    m.record_indexing(30)
    m.record_error("llm_error", "timeout", "/query")
    m.set_model_versions({"llm_model": "x"})
    snap = m.snapshot()
    c = snap["counters"]
    assert (c["query_count"], c["successful_queries"], c["failed_queries"]) == (3, 2, 1)
    assert (c["documents_ingested"], c["chunks_ingested"], c["ingestion_failures"]) == (2, 30, 1)
    assert c["chunks_indexed"] == 30 and c["indexing_runs"] == 1 and c["errors_total"] == 1
    assert snap["latency"]["retrieval"]["avg_ms"] == 15.0
    assert snap["latency"]["llm"]["count"] == 1 and snap["latency"]["llm"]["avg_ms"] == 100.0
    assert snap["latency"]["query_total"]["count"] == 2
    assert snap["errors_by_type"] == {"llm_error": 1} and snap["recent_errors"][0]["path"] == "/query"
    assert snap["model_versions"] == {"llm_model": "x"}
    json.dumps(snap)  # must be JSON serialisable


def test_recent_errors_are_bounded():
    m = MetricsRegistry(max_recent_errors=3)
    for i in range(10):
        m.record_error("e", f"message {i}")
    snap = m.snapshot()
    assert len(snap["recent_errors"]) == 3 and snap["counters"]["errors_total"] == 10


def test_registry_is_thread_safe():
    m = MetricsRegistry()
    threads = [threading.Thread(target=lambda: [m.record_request("/x", "GET", 200, 1.0) for _ in range(200)]) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert m.snapshot()["counters"]["total_requests"] == 1600


def test_jsonl_event_logger_appends_valid_json_lines(tmp_path):
    log = JsonlEventLogger(tmp_path / "nested" / "events.jsonl")
    log.log("query", latency_ms=12.5)
    log.log("request", status=200)
    lines = (tmp_path / "nested" / "events.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["event"] for r in records] == ["query", "request"]
    assert records[0]["latency_ms"] == 12.5 and "timestamp" in records[0]


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.request_id = "abc"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello world" and payload["request_id"] == "abc" and payload["level"] == "INFO"
