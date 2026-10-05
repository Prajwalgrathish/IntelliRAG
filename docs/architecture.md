# IntelliRAG architecture notes

## Data flow

```
Upload / data/documents  ->  loaders (TXT, PDF, DOCX)  ->  clean_text  ->  SentenceChunker
   -> data/processed/<doc_id>.jsonl (+ .meta.json)
   -> TransformerEncoder (mean pooling, L2 norm)  ->  FaissStore (IndexFlatIP) + metadata.jsonl
Query -> preprocess_query -> [IntentPredictor] -> Retriever (search, min_similarity, lexical rerank)
      -> build_context ([1]..[n]) -> prompt -> BaseLLM (local HF | OpenAI-compatible API)
      -> answer + parsed citations -> sources
```

## Design decisions

- **Ingest and index are separate steps.** Ingestion writes inspectable processed chunks; indexing embeds them. Changing the embedding model only needs `--rebuild`, not re-extraction.
- **Chunks never span pages/sections**, so page metadata in citations is exact.
- **Cosine similarity = inner product on L2-normalised vectors** in an exact `IndexFlatIP`; appropriate for portfolio-scale corpora. Swap in IVF/HNSW inside `FaissStore` for larger corpora.
- **No stop-word removal / stemming for transformer inputs.** Stop words are removed only for lexical helpers (keywords, reranking, evaluation).
- **The LLM is never called without context.** If no chunk passes `MIN_SIMILARITY`, the pipeline returns the fixed "insufficient information" answer.
- **Index/encoder compatibility is enforced** (model name and dimension recorded in the manifest).

## Neural component: query-intent classifier

A frozen pretrained transformer (`CLASSIFICATION_MODEL_NAME`) embeds the query; a small PyTorch MLP head is trained on `data/training/intent_queries.csv` (`text,label`). Predicted intent (a) widens `top_k` for `summary`/`comparison` questions when the caller did not set `top_k`, and (b) adds answer-style guidance to the prompt. If no checkpoint exists the pipeline works unchanged. The shipped CSV is a small seed set; replace/extend it with your own queries for meaningful results.

## Evaluation

- Retrieval: `evidence_phrases` / `relevant_documents` / `relevant_chunk_ids` define relevance -> Recall@K, Precision@K, HitRate@K, MRR, similarity statistics.
- Generation (no judge model needed): embedding answer/context relevance, sentence-level groundedness (lexical + embedding support), citation coverage and validity. These are heuristic proxies.
- Optional LLM-as-judge (`--llm-judge`) for faithfulness/relevance scores.
