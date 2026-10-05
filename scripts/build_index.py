"""CLI: embed processed chunks with the configured transformer and build/update the FAISS index."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys

from app.config import get_settings
from app.dependencies import ServiceContainer
from src.ingestion.service import load_processed_chunks
from src.utils.logging import configure_logging
from src.vectorstore.indexer import index_chunks


def main(argv=None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=settings.processed_dir)
    parser.add_argument("--vector-db-path", default=settings.vector_db_path)
    parser.add_argument("--embedding-model", default=settings.embedding_model_name)
    parser.add_argument("--rebuild", action="store_true", help="Clear the existing index and re-embed everything.")
    args = parser.parse_args(argv)

    settings = settings.model_copy(
        update={"processed_dir": args.processed_dir, "vector_db_path": args.vector_db_path, "embedding_model_name": args.embedding_model}
    )
    configure_logging(settings.log_level)
    chunks = load_processed_chunks(settings.processed_dir)
    if not chunks:
        print(json.dumps({"error": f"No processed chunks in '{settings.processed_dir}'. Run scripts/ingest_documents.py first."}))
        return 1

    container = ServiceContainer(settings)
    result = index_chunks(container.encoder, container.store, chunks, batch_size=settings.embedding_batch_size, rebuild=args.rebuild)
    container.persist_store()
    print(json.dumps({**result.__dict__, "embedding_model": container.encoder.model_name, "saved_to": settings.vector_db_path}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
