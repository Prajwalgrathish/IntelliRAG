"""Generation-quality evaluation (answer relevance, context relevance, groundedness, citations).

The default metrics are *heuristic proxies* computed with the embedding model and token overlap;
they need no extra model and are not a substitute for human review or an NLI/LLM judge.
An optional LLM-as-judge (``LLMJudge``) can be enabled explicitly.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

import numpy as np

from src.embeddings.transformer_encoder import TextEncoder
from src.evaluation.metrics import cosine_similarity, mean, token_f1
from src.llm.base import BaseLLM
from src.nlp.chunking import split_sentences
from src.nlp.preprocessing import content_tokens
from src.rag.context import parse_citations, strip_citations
from src.rag.prompts import is_abstention

_CITATION_GROUP = r"(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+"


def _split_answer_sentences(answer: str) -> list[str]:
    """Split an answer into sentences, keeping trailing citations with their sentence."""
    text = re.sub(rf"([.!?])\s*({_CITATION_GROUP})", lambda m: f" {m.group(2).strip()}{m.group(1)} ", answer)
    return split_sentences(text)


class GenerationEvaluator:
    def __init__(
        self, encoder: TextEncoder, lexical_threshold: float = 0.6, embedding_threshold: float = 0.75, min_sentence_tokens: int = 3
    ) -> None:
        self.encoder = encoder
        self.lexical_threshold = lexical_threshold
        self.embedding_threshold = embedding_threshold
        self.min_sentence_tokens = min_sentence_tokens

    def evaluate(self, question: str, answer: str, contexts: Sequence[str], reference: Optional[str] = None) -> dict[str, Any]:
        """Score one answer. ``contexts`` are the chunk texts in citation order ([1], [2], ...)."""
        abstained = is_abstention(answer)
        clean_answer = strip_citations(answer)
        q_vec = self.encoder.encode_queries([question])[0]
        ctx_vecs = self.encoder.encode_documents(list(contexts)) if contexts else np.zeros((0, len(q_vec)))

        result: dict[str, Any] = {
            "abstained": abstained,
            "context_relevance": mean([cosine_similarity(q_vec, v) for v in ctx_vecs]) if len(ctx_vecs) else None,
            "answer_relevance": cosine_similarity(q_vec, self.encoder.encode_documents([clean_answer])[0]) if clean_answer else None,
            "reference_f1": token_f1(clean_answer, reference) if reference else None,
            "groundedness": None, "citation_coverage": None, "citation_validity": None,
            "sentences_evaluated": 0,
        }
        if abstained or not contexts:
            return result

        sentences = [s for s in _split_answer_sentences(answer) if len(content_tokens(strip_citations(s))) >= self.min_sentence_tokens]
        result["sentences_evaluated"] = len(sentences)
        if sentences:
            ctx_tokens = set(content_tokens(" ".join(contexts)))
            sent_vecs = self.encoder.encode_documents([strip_citations(s) for s in sentences])
            supported = cited = 0
            for sentence, vec in zip(sentences, sent_vecs):
                tokens = set(content_tokens(strip_citations(sentence)))
                lexical = len(tokens & ctx_tokens) / len(tokens) if tokens else 0.0
                semantic = max(cosine_similarity(vec, c) for c in ctx_vecs)
                supported += int(lexical >= self.lexical_threshold or semantic >= self.embedding_threshold)
                cited += int(bool(parse_citations(sentence)))
            result["groundedness"] = supported / len(sentences)
            result["citation_coverage"] = cited / len(sentences)
        all_cited = parse_citations(answer)
        if all_cited:
            valid = [i for i in all_cited if 1 <= i <= len(contexts)]
            result["citation_validity"] = len(valid) / len(all_cited)
        return result


def summarize_generation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item generation metrics (means ignore items where a metric is undefined)."""
    keys = ["answer_relevance", "context_relevance", "groundedness", "citation_coverage", "citation_validity", "reference_f1"]
    summary: dict[str, Any] = {k: mean([r.get(k) for r in rows]) for k in keys}
    summary["num_items"] = len(rows)
    summary["abstention_rate"] = mean([1.0 if r.get("abstained") else 0.0 for r in rows])
    summary["mean_llm_ms"] = mean([r.get("llm_ms") for r in rows])
    summary["mean_total_ms"] = mean([r.get("total_ms") for r in rows])
    return summary


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of retrieval-augmented answers. Judge ONLY against the supplied context. "
    'Reply with a single JSON object: {"faithfulness": 1-5, "relevance": 1-5, "reason": "<one sentence>"}. '
    "faithfulness: are all claims supported by the context (5 = fully supported). "
    "relevance: does the answer address the question (5 = fully)."
)


class LLMJudge:
    """Optional LLM-as-judge. Slower, costs tokens, and inherits the judge model's biases."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def judge(self, question: str, answer: str, contexts: Sequence[str]) -> dict[str, Any]:
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
        prompt = f"Context:\n{numbered}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nJSON:"
        raw = self.llm.generate(JUDGE_SYSTEM_PROMPT, prompt).text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"error": "Judge did not return JSON", "raw": raw[:300]}
        try:
            data = json.loads(match.group(0))
            return {
                "faithfulness": float(data["faithfulness"]),
                "relevance": float(data["relevance"]),
                "reason": str(data.get("reason", "")),
            }
        except (ValueError, KeyError, TypeError) as exc:
            return {"error": f"Could not parse judge output: {exc}", "raw": raw[:300]}
