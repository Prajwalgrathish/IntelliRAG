import json

import httpx
import pytest

from src.evaluation.generation import GenerationEvaluator, LLMJudge, summarize_generation
from src.evaluation.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank, token_f1
from src.evaluation.retrieval import EvalItem, evaluate_retrieval, is_relevant_chunk, load_eval_set
from src.llm import APILLM, LLMError, create_llm
from src.nlp.chunking import Chunk
from src.rag.context import build_context, parse_citations, strip_citations
from src.rag.pipeline import RAGPipeline
from src.rag.prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, build_user_prompt, is_abstention
from src.retrieval import Retriever, RetrievedChunk
from src.vectorstore.faiss_store import FaissStore
from src.vectorstore.indexer import index_chunks
from tests.helpers import FakeLLM


def make_retrieved(texts, **chunk_kwargs):
    return [
        RetrievedChunk(
            Chunk(f"c{i}", f"doc{i}.pdf", t, f"/d/doc{i}.pdf", page=i + 1, **chunk_kwargs), vector_score=0.9 - i * 0.1, score=0.9 - i * 0.1, rank=i + 1
        )
        for i, t in enumerate(texts)
    ]


@pytest.fixture()
def pipeline_parts(encoder, sample_chunks):
    store = FaissStore()
    index_chunks(encoder, store, sample_chunks)
    return store, Retriever(encoder, store)


# ---------------------------------------------------------------- context & prompts
def test_context_is_numbered_with_source_headers():
    ctx = build_context(make_retrieved(["alpha text", "beta text"]))
    assert ctx.text.startswith("[1] (document: doc0.pdf, page: 1)\nalpha text")
    assert "[2] (document: doc1.pdf, page: 2)\nbeta text" in ctx.text
    assert len(ctx.used) == 2 and not ctx.truncated


def test_context_respects_word_budget_and_always_keeps_first_chunk():
    chunks = make_retrieved(["one two three four five", "six seven eight nine ten", "eleven twelve"])
    ctx = build_context(chunks, max_words=8)
    assert len(ctx.used) == 1 and ctx.truncated
    tiny = build_context(chunks, max_words=3)
    assert tiny.used[0].chunk.chunk_id == "c0" and tiny.word_count == 3


def test_parse_and_strip_citations():
    assert parse_citations("Fact [1]. Other fact [2, 3] and [1].") == [1, 2, 3, 1]
    assert strip_citations("Fact [1] here [2, 3].") == "Fact here ."


def test_prompts_enforce_grounding_rules():
    for phrase in ("ONLY", "Never invent sources", "Cite", "uncertain"):
        assert phrase in SYSTEM_PROMPT
    assert NO_CONTEXT_ANSWER in SYSTEM_PROMPT
    prompt = build_user_prompt("What is X?", "[1] context", intent="summary")
    assert "[1] context" in prompt and "What is X?" in prompt and "Question type: summary" in prompt
    assert "Question type" not in build_user_prompt("What is X?", "[1] ctx", intent=None)


def test_abstention_detection():
    assert is_abstention(NO_CONTEXT_ANSWER)
    assert is_abstention("  " + NO_CONTEXT_ANSWER.upper() + " The context mentions tides.")
    assert not is_abstention("Plants use chlorophyll [1].")


# ---------------------------------------------------------------- pipeline
def test_pipeline_returns_grounded_answer_with_cited_sources(pipeline_parts, llm):
    _, retriever = pipeline_parts
    result = RAGPipeline(retriever, llm, default_top_k=2).answer("How does photosynthesis work in plants?")
    assert result.llm_called and result.answer == llm.reply
    assert result.sources[0].document == "plants.txt" and result.sources[0].cited
    assert result.retrieved_chunks == len(result.sources) == 2
    assert result.sources[1].cited is False
    system_prompt, user_prompt = llm.calls[0]
    assert system_prompt == SYSTEM_PROMPT and "photosynthesis" in user_prompt.lower() and "[1]" in user_prompt
    assert result.total_ms >= result.retrieval_ms >= 0 and result.llm_ms == 1.5


def test_pipeline_never_calls_llm_without_context(encoder, sample_chunks, llm):
    store = FaissStore()
    index_chunks(encoder, store, sample_chunks)
    pipeline = RAGPipeline(Retriever(encoder, store, min_similarity=0.95), llm)
    result = pipeline.answer("quantum chromodynamics lattice gauge theory")
    assert result.answer == NO_CONTEXT_ANSWER and result.sources == []
    assert not result.llm_called and llm.calls == []


def test_pipeline_explicit_top_k_beats_intent_adjustment(pipeline_parts, llm):
    class SummaryIntent:
        def predict(self, texts):
            from src.neural.model import IntentPrediction

            return [IntentPrediction("summary", 0.9, {"summary": 0.9})]

    _, retriever = pipeline_parts
    pipeline = RAGPipeline(retriever, llm, default_top_k=2, intent_predictor=SummaryIntent())
    auto = pipeline.answer("Summarize carbon dioxide")
    explicit = pipeline.answer("Summarize carbon dioxide", top_k=1)
    assert auto.top_k_used == 4 and auto.intent == "summary"
    assert explicit.top_k_used == 1 and len(explicit.sources) == 1
    assert "Question type: summary" in llm.calls[0][1]


def test_pipeline_survives_intent_classifier_failure(pipeline_parts, llm):
    class Broken:
        def predict(self, texts):
            raise RuntimeError("boom")

    _, retriever = pipeline_parts
    result = RAGPipeline(retriever, llm, intent_predictor=Broken()).answer("photosynthesis in plants")
    assert result.intent is None and result.llm_called


def test_pipeline_rejects_empty_question(pipeline_parts, llm):
    with pytest.raises(ValueError):
        RAGPipeline(pipeline_parts[1], llm).answer("   ")


# ---------------------------------------------------------------- LLM backends
def _api_llm(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return APILLM("test-model", "https://llm.example/v1", "secret-key", client=client, **kwargs)


def test_api_llm_sends_openai_style_request_and_parses_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "test-model", "choices": [{"message": {"content": " Answer [1] "}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    response = _api_llm(handler, max_new_tokens=50, temperature=0.1).generate("sys", "user")
    assert response.text == "Answer [1]" and response.prompt_tokens == 7 and response.completion_tokens == 3
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["max_tokens"] == 50


def test_api_llm_raises_llm_error_on_http_error_and_bad_payload():
    with pytest.raises(LLMError, match="HTTP 401"):
        _api_llm(lambda r: httpx.Response(401, text="unauthorized")).generate("s", "u")
    with pytest.raises(LLMError, match="Unexpected"):
        _api_llm(lambda r: httpx.Response(200, json={"nope": 1})).generate("s", "u")


def test_api_llm_requires_key_and_factory_validates_provider():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        APILLM("m", "https://x/v1", api_key=None)
    with pytest.raises(LLMError, match="Unknown"):
        create_llm("carrier-pigeon", "m")
    assert create_llm("local", "some/model").provider == "local"  # lazy: nothing is downloaded here
    assert create_llm("api", "gpt", api_base_url="https://x/v1", api_key="k").provider == "api"


# ---------------------------------------------------------------- evaluation
def test_retrieval_metric_functions():
    flags = [False, True, False, True]
    assert recall_at_k(flags, total_relevant=2, k=2) == 0.5
    assert recall_at_k(flags, total_relevant=2, k=4) == 1.0
    assert precision_at_k(flags, 4) == 0.5 and precision_at_k(flags, 1) == 0.0
    assert hit_at_k(flags, 1) == 0.0 and hit_at_k(flags, 2) == 1.0
    assert reciprocal_rank(flags) == 0.5 and reciprocal_rank([False]) == 0.0
    assert token_f1("plants use chlorophyll", "plants use chlorophyll") == 1.0
    assert token_f1("tides", "chlorophyll") == 0.0


def test_relevance_predicate_variants(sample_chunks):
    plants = sample_chunks[0]
    assert is_relevant_chunk(plants, EvalItem("1", "q", evidence_phrases=["Using   CHLOROPHYLL"]))
    assert not is_relevant_chunk(plants, EvalItem("1", "q", evidence_phrases=["chlorophyll"], relevant_documents=["other.txt"]))
    assert is_relevant_chunk(plants, EvalItem("1", "q", relevant_documents=["plants.txt"]))
    assert is_relevant_chunk(plants, EvalItem("1", "q", relevant_chunk_ids=["plants.txt-0000"]))
    assert not is_relevant_chunk(plants, EvalItem("1", "q", evidence_phrases=["diesel"]))


def test_evaluate_retrieval_computes_real_metrics(pipeline_parts):
    store, retriever = pipeline_parts
    items = [
        EvalItem("a", "How do plants make glucose from sunlight?", evidence_phrases=["chlorophyll in the leaves"]),
        EvalItem("b", "What causes ocean tides?", evidence_phrases=["gravitational pull of the moon"]),
        EvalItem("c", "Unanswerable", evidence_phrases=["text that is not in any chunk"]),
    ]
    report = evaluate_retrieval(retriever, store, items, ks=[1, 3])
    agg = report["aggregate"]
    assert agg["num_items_evaluated"] == 2 and agg["num_items_skipped"] == 1
    assert report["skipped_items"][0]["id"] == "c"
    assert agg["recall@1"] == 1.0 and agg["precision@1"] == 1.0 and agg["mrr"] == 1.0
    assert agg["precision@3"] == pytest.approx(1 / 3)
    assert 0 < agg["mean_top1_vector_score"] <= 1.0


def test_load_eval_set_validates_input(tmp_path):
    good = tmp_path / "eval.jsonl"
    good.write_text(json.dumps({"question": "Q?", "evidence_phrases": ["x"]}) + "\n\n")
    assert load_eval_set(good)[0].id == "q1"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "Q?"}) + "\n")
    with pytest.raises(ValueError, match="at least one of"):
        load_eval_set(bad)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_eval_set(broken)
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "missing.jsonl")


def test_generation_evaluator_distinguishes_grounded_from_ungrounded(encoder):
    contexts = ["Photosynthesis lets plants convert sunlight, water and carbon dioxide into glucose using chlorophyll."]
    evaluator = GenerationEvaluator(encoder)
    grounded = evaluator.evaluate(
        "How do plants make glucose?", "Plants convert sunlight, water and carbon dioxide into glucose using chlorophyll [1].", contexts
    )
    invented = evaluator.evaluate(
        "How do plants make glucose?", "Volcanic eruptions release molten basalt across Iceland.", contexts
    )
    assert grounded["groundedness"] == 1.0 and grounded["citation_coverage"] == 1.0 and grounded["citation_validity"] == 1.0
    assert invented["groundedness"] == 0.0 and invented["citation_coverage"] == 0.0 and invented["citation_validity"] is None
    assert grounded["answer_relevance"] > invented["answer_relevance"]
    assert grounded["context_relevance"] > 0


def test_generation_evaluator_flags_invalid_citations_and_abstentions(encoder):
    contexts = ["Tides come from the moon's gravity acting on seawater."]
    evaluator = GenerationEvaluator(encoder)
    wrong = evaluator.evaluate("What causes tides?", "Tides come from the moon's gravity acting on seawater [5].", contexts)
    assert wrong["citation_validity"] == 0.0
    refusal = evaluator.evaluate("What causes tides?", NO_CONTEXT_ANSWER, contexts)
    assert refusal["abstained"] and refusal["groundedness"] is None


def test_summarize_generation_ignores_undefined_metrics():
    summary = summarize_generation([
        {"groundedness": 1.0, "abstained": False, "llm_ms": 10},
        {"groundedness": None, "abstained": True, "llm_ms": 0},
        {"groundedness": 0.5, "abstained": False, "llm_ms": 20},
    ])
    assert summary["groundedness"] == 0.75 and summary["abstention_rate"] == pytest.approx(1 / 3)
    assert summary["answer_relevance"] is None and summary["num_items"] == 3


def test_llm_judge_parses_json_and_reports_failures():
    ok = LLMJudge(FakeLLM('Sure: {"faithfulness": 4, "relevance": 5, "reason": "ok"}')).judge("q", "a", ["c"])
    assert ok == {"faithfulness": 4.0, "relevance": 5.0, "reason": "ok"}
    assert "error" in LLMJudge(FakeLLM("no json here")).judge("q", "a", ["c"])
