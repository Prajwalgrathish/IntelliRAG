"""Retrieval evaluation against a labelled evaluation set.

Each JSONL line describes a question and how to recognise relevant chunks:
  {"id": "q1", "question": "...", "evidence_phrases": ["exact phrase"], "relevant_documents": ["file.txt"],
   "relevant_chunk_ids": ["..."], "reference_answer": "..."}
A chunk is relevant if its id is listed, or if it contains an evidence phrase (restricted to the
listed documents when given), or - when only documents are given - if it belongs to one of them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from src.evaluation.metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from src.nlp.chunking import Chunk
from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FaissStore


@dataclass
class EvalItem:
    id: str
    question: str
    reference_answer: Optional[str] = None
    evidence_phrases: list[str] = field(default_factory=list)
    relevant_documents: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_eval_set(path: Union[str, Path]) -> list[EvalItem]:
    """Load and validate an evaluation set (JSONL)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    items: list[EvalItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no}: invalid JSON ({exc})") from exc
        if not raw.get("question"):
            raise ValueError(f"{path.name}:{line_no}: 'question' is required")
        item = EvalItem(
            id=str(raw.get("id", f"q{len(items) + 1}")),
            question=raw["question"],
            reference_answer=raw.get("reference_answer"),
            evidence_phrases=list(raw.get("evidence_phrases", [])),
            relevant_documents=list(raw.get("relevant_documents", [])),
            relevant_chunk_ids=list(raw.get("relevant_chunk_ids", [])),
        )
        if not (item.evidence_phrases or item.relevant_documents or item.relevant_chunk_ids):
            raise ValueError(
                f"{path.name}:{line_no}: provide at least one of evidence_phrases, relevant_documents, relevant_chunk_ids"
            )
        items.append(item)
    if not items:
        raise ValueError(f"{path.name} contains no evaluation items.")
    return items


def is_relevant_chunk(chunk: Chunk, item: EvalItem) -> bool:
    """Ground-truth relevance predicate (see module docstring)."""
    if chunk.chunk_id in item.relevant_chunk_ids:
        return True
    doc_ok = not item.relevant_documents or chunk.document in item.relevant_documents
    if item.evidence_phrases:
        text = _norm(chunk.text)
        return doc_ok and any(_norm(p) in text for p in item.evidence_phrases)
    if item.relevant_documents:
        return doc_ok
    return False


def evaluate_retrieval(
    retriever: Retriever, store: FaissStore, items: Sequence[EvalItem], ks: Sequence[int] = (1, 3, 5)
) -> dict[str, Any]:
    """Compute Recall@K, Precision@K, HitRate@K, MRR and similarity statistics."""
    ks = sorted(set(ks))
    corpus = store.all_chunks()
    per_item: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in items:
        total_relevant = sum(1 for c in corpus if is_relevant_chunk(c, item))
        if total_relevant == 0:
            skipped.append({"id": item.id, "reason": "No chunk in the index matches the item's relevance criteria."})
            continue
        result = retriever.retrieve(item.question, top_k=max(ks))
        flags = [is_relevant_chunk(r.chunk, item) for r in result.chunks]
        row: dict[str, Any] = {
            "id": item.id,
            "question": item.question,
            "relevant_chunks_in_corpus": total_relevant,
            "retrieved": len(result.chunks),
            "relevance_flags": flags,
            "top_vector_score": result.chunks[0].vector_score if result.chunks else None,
            "mean_vector_score": mean([r.vector_score for r in result.chunks]),
            "reciprocal_rank": reciprocal_rank(flags),
            "retrieval_ms": round(result.total_ms, 3),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(flags, total_relevant, k)
            row[f"precision@{k}"] = precision_at_k(flags, k)
            row[f"hit@{k}"] = hit_at_k(flags, k)
        per_item.append(row)

    aggregate: dict[str, Any] = {"num_items_evaluated": len(per_item), "num_items_skipped": len(skipped)}
    for k in ks:
        aggregate[f"recall@{k}"] = mean([r[f"recall@{k}"] for r in per_item])
        aggregate[f"precision@{k}"] = mean([r[f"precision@{k}"] for r in per_item])
        aggregate[f"hit_rate@{k}"] = mean([r[f"hit@{k}"] for r in per_item])
    aggregate["mrr"] = mean([r["reciprocal_rank"] for r in per_item])
    aggregate["mean_top1_vector_score"] = mean([r["top_vector_score"] for r in per_item])
    aggregate["mean_retrieval_ms"] = mean([r["retrieval_ms"] for r in per_item])
    return {"ks": ks, "aggregate": aggregate, "per_item": per_item, "skipped_items": skipped}
