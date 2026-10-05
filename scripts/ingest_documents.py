"""CLI: ingest every supported document in a directory (extract -> clean -> chunk -> processed JSONL)."""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path)

import argparse
import json
import sys

from app.config import get_settings
from src.ingestion.base import DocumentLoadError
from src.ingestion.service import ingest_directory
from src.nlp.chunking import SentenceChunker
from src.utils.logging import configure_logging


def main(argv=None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=settings.documents_dir)
    parser.add_argument("--processed-dir", default=settings.processed_dir)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="Words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap, help="Words of overlap.")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(settings.log_level)
    try:
        chunker = SentenceChunker(args.chunk_size, args.chunk_overlap)
        results, errors = ingest_directory(args.input_dir, args.processed_dir, chunker, recursive=not args.no_recursive)
    except (DocumentLoadError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    summary = {
        "ingested": [r.__dict__ for r in results],
        "failed": errors,
        "total_documents": len(results),
        "total_chunks": sum(r.chunks for r in results),
    }
    print(json.dumps(summary, indent=2))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
