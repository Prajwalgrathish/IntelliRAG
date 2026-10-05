"""CLI: evaluate retrieval (Recall@K, Precision@K, MRR) and generation (relevance, groundedness, citations)."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.dependencies import ServiceContainer
from src.evaluation.generation import GenerationEvaluator, LLMJudge, summarize_generation
from src.evaluation.retrieval import evaluate_retrieval, load_eval_set
from src.utils.logging import configure_logging


def main(argv=None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default="data/evaluation/eval_set.example.jsonl")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--output-dir", default=settings.evaluation_dir)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--skip-generation", action="store_true", help="Retrieval metrics only (no LLM needed).")
    parser.add_argument("--llm-judge", action="store_true", help="Also run the optional LLM-as-judge (extra LLM calls).")
    args = parser.parse_args(argv)

    configure_logging(settings.log_level)
    container = ServiceContainer(settings)
    store = container.store
    if store.is_empty:
        print(json.dumps({"error": "Vector index is empty. Run scripts/ingest_documents.py and scripts/build_index.py first."}))
        return 2
    items = load_eval_set(args.eval_file)[: args.max_items]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    config = {
        "generated_at": stamp, "eval_file": args.eval_file, "num_eval_items": len(items),
        "embedding_model": settings.embedding_model_name, "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model_name, "top_k": args.top_k, "min_similarity": settings.min_similarity,
        "rerank_enabled": settings.rerank_enabled, "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap, "indexed_chunks": len(store),
    }

    retrieval_report = {"config": config, **evaluate_retrieval(container.retriever, store, items, ks=args.ks)}
    (out_dir / "retrieval_report.json").write_text(json.dumps(retrieval_report, indent=2, default=float), encoding="utf-8")
    summary = {"config": config, "retrieval": retrieval_report["aggregate"]}

    if not args.skip_generation:
        evaluator = GenerationEvaluator(container.encoder)
        judge = LLMJudge(container.llm) if args.llm_judge else None
        rows, answers = [], []
        for item in items:
            try:
                result = container.pipeline.answer(item.question, top_k=args.top_k)
            except Exception as exc:  # record and continue
                rows.append({"id": item.id, "error": f"{exc.__class__.__name__}: {exc}"})
                continue
            contexts = [s.text for s in result.sources]
            metrics = evaluator.evaluate(item.question, result.answer, contexts, item.reference_answer)
            row = {"id": item.id, "question": item.question, "answer": result.answer, "llm_ms": result.llm_ms,
                   "total_ms": result.total_ms, "sources": [s.chunk_id for s in result.sources], **metrics}
            if judge and result.llm_called:
                row["llm_judge"] = judge.judge(item.question, result.answer, contexts)
            rows.append(row)
            answers.append({"id": item.id, "question": item.question, "answer": result.answer})
        ok_rows = [r for r in rows if "error" not in r]
        generation_report = {
            "config": {**config, "llm_judge_enabled": bool(judge)},
            "aggregate": summarize_generation(ok_rows),
            "errors": [r for r in rows if "error" in r],
            "per_item": rows,
            "notes": "Groundedness/relevance are embedding+lexical heuristics, not human or NLI judgements.",
        }
        if judge:
            scored = [r["llm_judge"] for r in ok_rows if "llm_judge" in r and "error" not in r["llm_judge"]]
            generation_report["aggregate"]["llm_judge_faithfulness"] = (
                sum(s["faithfulness"] for s in scored) / len(scored) if scored else None
            )
            generation_report["aggregate"]["llm_judge_relevance"] = (
                sum(s["relevance"] for s in scored) / len(scored) if scored else None
            )
        (out_dir / "generation_report.json").write_text(json.dumps(generation_report, indent=2, default=float), encoding="utf-8")
        summary["generation"] = generation_report["aggregate"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nReports written to: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
