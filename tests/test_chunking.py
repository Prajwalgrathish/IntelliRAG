import pytest

from src.ingestion.base import Segment
from src.nlp.chunking import Chunk, SentenceChunker, split_sentences


def test_split_sentences_handles_abbreviations_and_paragraphs():
    text = "Dr. Smith wrote the report. It was long, e.g. 200 pages. Was it good? Yes!\n\nNew paragraph here."
    assert split_sentences(text) == [
        "Dr. Smith wrote the report.",
        "It was long, e.g. 200 pages.",
        "Was it good?",
        "Yes!",
        "New paragraph here.",
    ]


def test_chunks_never_exceed_chunk_size():
    text = " ".join(f"This is sentence {i} about retrieval." for i in range(60))
    chunker = SentenceChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 40 for c in chunks)


def test_consecutive_chunks_overlap_by_whole_sentences():
    text = " ".join(f"Sentence {i} has exactly six words." for i in range(30))
    chunks = SentenceChunker(chunk_size=30, chunk_overlap=12).chunk_text(text)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(prev)[-1]
        assert last_sentence in nxt


def test_all_content_is_preserved_across_chunks():
    sentences = [f"Fact number {i} is recorded here." for i in range(25)]
    chunks = SentenceChunker(chunk_size=20, chunk_overlap=0).chunk_text(" ".join(sentences))
    assert " ".join(chunks) == " ".join(sentences)


def test_oversized_sentence_is_split_into_word_windows():
    sentence = " ".join(f"w{i}" for i in range(100)) + "."
    chunks = SentenceChunker(chunk_size=30, chunk_overlap=5).chunk_text(sentence)
    assert len(chunks) >= 4
    assert all(len(c.split()) <= 30 for c in chunks)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=0, chunk_overlap=0)


def test_chunk_document_attaches_metadata_per_page():
    segments = [Segment("Page one text. More page one text.", page=1), Segment("Page two text.", page=2, section="Results")]
    chunks = SentenceChunker(50, 5).chunk_document(doc_id="doc-abc", document_name="doc.pdf", source_path="/x/doc.pdf", segments=segments)
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[1].section == "Results"
    assert chunks[0].chunk_id == "doc-abc-0000" and chunks[1].chunk_id == "doc-abc-0001"
    assert chunks[0].keywords and chunks[0].word_count == len(chunks[0].text.split())


def test_chunk_dict_roundtrip_ignores_unknown_fields():
    chunk = Chunk(chunk_id="a", document="d", text="t", source_path="p", page=2, keywords=["k"])
    data = chunk.to_dict() | {"unexpected": 1}
    assert Chunk.from_dict(data) == chunk
