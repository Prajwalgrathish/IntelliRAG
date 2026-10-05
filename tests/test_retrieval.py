import numpy as np
import pytest

from src.nlp.chunking import Chunk
from src.retrieval import LexicalReranker, Retriever, RetrievedChunk
from src.vectorstore.faiss_store import FaissStore, VectorStoreError
from src.vectorstore.indexer import index_chunks


def build_store(encoder, chunks):
    store = FaissStore()
    index_chunks(encoder, store, chunks)
    return store


def test_store_add_and_search_returns_best_match_first(encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    hits = store.search(encoder.encode_queries(["how do plants use sunlight for photosynthesis"])[0], k=3)
    assert hits[0][0].document == "plants.txt"
    assert hits[0][1] >= hits[1][1] >= hits[2][1]
    assert len(store) == 4 and store.dimension == encoder.dimension


def test_duplicate_chunk_ids_are_not_added_twice(encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    result = index_chunks(encoder, store, sample_chunks)
    assert result.new_chunks == 0 and result.skipped_existing == 4
    assert len(store) == 4


def test_dimension_mismatch_is_rejected(encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    with pytest.raises(VectorStoreError):
        store.add(np.ones((1, 7), dtype=np.float32), [Chunk("x", "d", "t", "p")])
    with pytest.raises(VectorStoreError):
        store.search(np.ones(7, dtype=np.float32), 1)


def test_save_and_load_roundtrip_preserves_results_and_metadata(tmp_path, encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    store.save(tmp_path / "vs")
    assert FaissStore.exists(tmp_path / "vs")
    loaded = FaissStore.load(tmp_path / "vs")

    query = encoder.encode_queries(["gravitational pull of the moon on ocean tides"])[0]
    original, restored = store.search(query, 2), loaded.search(query, 2)
    assert [c.chunk_id for c, _ in original] == [c.chunk_id for c, _ in restored]
    assert np.allclose([s for _, s in original], [s for _, s in restored])
    assert restored[0][0].source_path == "/data/oceans.txt"
    assert loaded.embedding_model_name == encoder.model_name


def test_load_detects_corrupted_store(tmp_path, encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    store.save(tmp_path / "vs")
    meta = tmp_path / "vs" / FaissStore.META_FILE
    meta.write_text("\n".join(meta.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(VectorStoreError):
        FaissStore.load(tmp_path / "vs")
    with pytest.raises(VectorStoreError):
        FaissStore.load(tmp_path / "missing")


def test_assert_compatible_detects_model_change(encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    store.assert_compatible(encoder.model_name, encoder.dimension)
    with pytest.raises(VectorStoreError):
        store.assert_compatible("another-model", encoder.dimension)


def test_rebuild_clears_previous_vectors(encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    result = index_chunks(encoder, store, sample_chunks[:2], rebuild=True)
    assert result.total_chunks == 2 and len(store) == 2


def test_retriever_returns_ranked_chunks_with_metadata(encoder, sample_chunks):
    retriever = Retriever(encoder, build_store(encoder, sample_chunks))
    result = retriever.retrieve("What makes bread rise? yeast fermentation", top_k=2)
    assert [r.rank for r in result.chunks] == [1, 2]
    assert result.chunks[0].chunk.document == "bakery.txt"
    assert result.total_ms > 0 and result.embedding_ms >= 0


def test_top_k_is_respected_and_validated(encoder, sample_chunks):
    retriever = Retriever(encoder, build_store(encoder, sample_chunks))
    assert len(retriever.retrieve("carbon dioxide", top_k=1).chunks) == 1
    assert len(retriever.retrieve("carbon dioxide", top_k=10).chunks) == 4
    with pytest.raises(ValueError):
        retriever.retrieve("carbon dioxide", top_k=0)


def test_min_similarity_filters_irrelevant_chunks(encoder, sample_chunks):
    store = build_store(encoder, sample_chunks)
    strict = Retriever(encoder, store, min_similarity=0.9)
    assert strict.retrieve("quantum chromodynamics lattice gauge", top_k=3).chunks == []


def test_retrieve_on_empty_store_returns_nothing(encoder):
    assert Retriever(encoder, FaissStore()).retrieve("anything at all").chunks == []


def test_reranker_promotes_lexically_matching_chunk(sample_chunks):
    exact = RetrievedChunk(sample_chunks[0], vector_score=0.50, score=0.50)  # plants: matches query terms
    other = RetrievedChunk(sample_chunks[1], vector_score=0.60, score=0.60)  # engines: no overlap
    ranked = LexicalReranker(alpha=0.5).rerank("photosynthesis chlorophyll leaves", [other, exact])
    assert ranked[0].chunk.document == "plants.txt"
    assert ranked[0].lexical_overlap == pytest.approx(1.0)
    assert ranked[0].vector_score == 0.50


def test_reranker_validates_alpha():
    with pytest.raises(ValueError):
        LexicalReranker(alpha=1.5)
