import io

import pytest

from src.rag.prompts import NO_CONTEXT_ANSWER

PLANTS = (
    "Photosynthesis lets plants convert sunlight, water and carbon dioxide into glucose using chlorophyll. "
    "The process takes place mostly in the leaves of the plant. Oxygen is released as a by-product."
)
OCEANS = "Ocean tides are caused mainly by the gravitational pull of the moon on seawater. Spring tides occur at new and full moon."


def upload(client, name, content, mime="text/plain"):
    return client.post("/documents/ingest", files=[("files", (name, io.BytesIO(content.encode()), mime))])


def ingest_and_index(client):
    assert upload(client, "plants.txt", PLANTS).status_code == 200
    assert upload(client, "oceans.txt", OCEANS).status_code == 200
    response = client.post("/documents/index", json={})
    assert response.status_code == 200
    return response.json()


def test_health_reports_empty_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["index_loaded"] is False and body["indexed_chunks"] == 0
    assert body["uptime_seconds"] >= 0


def test_ingest_stores_file_and_returns_chunk_counts(client, settings):
    response = upload(client, "plants.txt", PLANTS)
    body = response.json()
    assert response.status_code == 200
    assert body["ingested"][0]["document"] == "plants.txt" and body["ingested"][0]["chunks"] >= 1
    assert body["total_chunks"] == body["ingested"][0]["chunks"] and body["failed"] == []
    from pathlib import Path

    assert Path(settings.documents_dir, "plants.txt").is_file()
    assert client.get("/documents").json()["documents"][0]["indexed_chunks"] == 0


def test_ingest_mixed_batch_reports_partial_failure(client):
    files = [
        ("files", ("good.txt", io.BytesIO(PLANTS.encode()), "text/plain")),
        ("files", ("broken.pdf", io.BytesIO(b"not a pdf"), "application/pdf")),
    ]
    response = client.post("/documents/ingest", files=files)
    body = response.json()
    assert response.status_code == 200
    assert [d["document"] for d in body["ingested"]] == ["good.txt"]
    assert body["failed"][0]["code"] == "corrupt_pdf"


def test_ingest_all_failures_returns_structured_422(client, settings):
    response = upload(client, "malware.exe", "MZ")
    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "ingestion_failed"
    assert body["error"]["details"]["failed"][0]["code"] == "unsupported_type"
    from pathlib import Path

    assert not Path(settings.documents_dir, "malware.exe").exists()


def test_failed_upload_does_not_destroy_existing_file(client, settings):
    from pathlib import Path

    assert upload(client, "keep.txt", PLANTS).status_code == 200
    original = Path(settings.documents_dir, "keep.txt").read_bytes()
    empty = client.post("/documents/ingest", files=[("files", ("keep.txt", io.BytesIO(b""), "text/plain"))])
    assert empty.status_code == 422
    assert Path(settings.documents_dir, "keep.txt").read_bytes() == original


def test_index_without_documents_is_409(client):
    response = client.post("/documents/index", json={})
    assert response.status_code == 409 and response.json()["error"]["code"] == "nothing_to_index"


def test_index_builds_persists_and_lists_documents(client, settings):
    from pathlib import Path

    body = ingest_and_index(client)
    assert body["new_chunks"] >= 2 and body["documents"] == 2 and body["embedding_model"] == "hashing-test-encoder"
    assert (Path(settings.vector_db_path) / "index.faiss").is_file()
    docs = client.get("/documents").json()
    assert docs["total_documents"] == 2 and docs["total_indexed_chunks"] == body["total_chunks"]
    assert all(d["indexed_chunks"] == d["chunks"] for d in docs["documents"])
    again = client.post("/documents/index", json={}).json()
    assert again["new_chunks"] == 0 and again["skipped_existing"] == body["total_chunks"]


def test_index_rejects_rebuild_with_document_ids(client):
    response = client.post("/documents/index", json={"rebuild": True, "document_ids": ["x"]})
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


def test_query_before_indexing_returns_409(client):
    response = client.post("/query", json={"question": "What is photosynthesis?"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "index_empty"


def test_query_returns_grounded_answer_with_real_sources(client, llm):
    ingest_and_index(client)
    response = client.post("/query", json={"question": "How do plants convert sunlight into glucose?", "top_k": 2})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == llm.reply and body["model"] == "fake-llm"
    assert body["retrieved_chunks"] == len(body["sources"]) == 2
    top = body["sources"][0]
    assert top["document"] == "plants.txt" and top["cited"] is True and top["citation_index"] == 1
    assert 0 < top["score"] <= 1.0 and top["chunk_id"].startswith("plants-")
    assert body["latency_ms"] >= body["retrieval_latency_ms"] >= 0 and body["llm_latency_ms"] == 1.5
    assert response.headers["X-Request-ID"]
    assert "Question:" in llm.calls[0][1] and "photosynthesis" in llm.calls[0][1].lower()


def test_query_with_irrelevant_question_abstains_without_calling_llm(settings, llm, encoder):
    from fastapi.testclient import TestClient

    from app.dependencies import ServiceContainer
    from app.main import create_app

    strict = settings.model_copy(update={"min_similarity": 0.9})
    app = create_app(strict, ServiceContainer(strict, encoder=encoder, llm=llm, intent_predictor=None))
    with TestClient(app) as c:
        ingest_and_index(c)
        body = c.post("/query", json={"question": "Explain quantum chromodynamics lattice gauge theory"}).json()
    assert body["answer"] == NO_CONTEXT_ANSWER and body["sources"] == [] and body["llm_called"] is False
    assert llm.calls == []


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "hi"}, {"question": "   ab  "}, {"question": "valid question", "top_k": 0},
     {"question": "valid question", "top_k": 51}, {"question": "valid question", "top_k": "many"}, {"question": "x" * 2001}],
)
def test_query_validation_errors_are_structured(client, payload):
    response = client.post("/query", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error" and isinstance(error["details"], list)


def test_llm_failure_returns_502_and_is_recorded_in_metrics(settings, encoder):
    from fastapi.testclient import TestClient

    from app.dependencies import ServiceContainer
    from app.main import create_app
    from src.llm.base import BaseLLM, LLMError

    class FailingLLM(BaseLLM):
        provider, model_name = "fail", "failing-llm"

        def generate(self, system_prompt, user_prompt):
            raise LLMError("upstream timeout")

    app = create_app(settings, ServiceContainer(settings, encoder=encoder, llm=FailingLLM(), intent_predictor=None))
    with TestClient(app) as c:
        ingest_and_index(c)
        response = c.post("/query", json={"question": "How do plants convert sunlight?"})
        metrics = c.get("/metrics").json()
    assert response.status_code == 502 and response.json()["error"]["code"] == "llm_error"
    assert metrics["counters"]["failed_queries"] == 1 and metrics["errors_by_type"]["llm_error"] == 1


def test_model_info_does_not_fabricate_values(client, settings):
    info = client.get("/model/info").json()
    assert info["embedding"]["model_name"] == settings.embedding_model_name
    assert info["classification"]["intent_classifier_loaded"] is False
    assert info["llm"]["model_name"] == "fake-llm" and info["llm"]["provider"] == "fake"
    assert info["retrieval"]["chunk_size"] == settings.chunk_size
    assert info["vector_index"]["chunks"] == 0
    ingest_and_index(client)
    assert client.get("/model/info").json()["vector_index"]["chunks"] > 0


def test_metrics_reflect_actual_traffic_and_event_log_is_written(client, settings):
    from pathlib import Path
    import json

    ingest_and_index(client)
    client.post("/query", json={"question": "What causes ocean tides?"})
    client.post("/query", json={"question": "x"})  # validation failure
    m = client.get("/metrics").json()
    c = m["counters"]
    assert c["documents_ingested"] == 2 and c["chunks_indexed"] >= 2
    assert c["query_count"] == 1 and c["successful_queries"] == 1
    assert c["failed_requests"] >= 1 and c["successful_requests"] >= 4
    assert m["latency"]["retrieval"]["count"] == 1 and m["latency"]["llm"]["count"] == 1
    assert m["model_versions"]["embedding_model"] == settings.embedding_model_name
    events = [json.loads(l) for l in Path(settings.logs_dir, "requests.jsonl").read_text().splitlines()]
    assert {"request", "query", "ingest", "index"} <= {e["event"] for e in events}


def test_unknown_route_returns_structured_error(client):
    response = client.get("/nope")
    assert response.status_code == 404 and response.json()["error"]["code"] == "http_error"
