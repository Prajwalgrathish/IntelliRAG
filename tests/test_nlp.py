import pytest

from src.nlp.cleaning import clean_text, fix_hyphenation, normalize_unicode, normalize_whitespace, remove_page_artifacts
from src.nlp.preprocessing import content_tokens, extract_keywords, preprocess_query


def test_normalize_unicode_fixes_ligatures_and_quotes():
    assert normalize_unicode("The \ufb01rst \u201cquote\u201d") == 'The first "quote"'


def test_fix_hyphenation_rejoins_split_words():
    assert fix_hyphenation("inter-\nnational trade") == "international trade"
    assert fix_hyphenation("well-known") == "well-known"


def test_remove_page_artifacts_only_drops_number_only_lines():
    text = "Intro text\n12\nPage 3 of 10\nIn 2023 revenue grew"
    cleaned = remove_page_artifacts(text)
    assert "Page 3 of 10" not in cleaned
    assert "In 2023 revenue grew" in cleaned


def test_normalize_whitespace_preserves_paragraphs():
    assert normalize_whitespace("a   b\t c\n\n\n\nd  \n e") == "a b c\n\nd\ne"


def test_clean_text_keeps_case_stopwords_and_negation():
    cleaned = clean_text("The system is NOT\u00a0reliable.\x00  Really   not.")
    assert cleaned == "The system is NOT reliable. Really not."


def test_preprocess_query_normalises_and_rejects_empty():
    assert preprocess_query("  What   is\nRAG?\u200b ") == "What is RAG?"
    with pytest.raises(ValueError):
        preprocess_query("   \n ")


def test_preprocess_query_truncates_long_input():
    assert len(preprocess_query("word " * 1000, max_chars=100)) <= 100


def test_content_tokens_drop_stopwords_only_for_lexical_utilities():
    tokens = content_tokens("The quick brown fox is on the mat")
    assert "the" not in tokens and "quick" in tokens and "mat" in tokens


def test_extract_keywords_orders_by_frequency_then_first_occurrence():
    text = "vector search uses embeddings. embeddings power vector search. search is fast."
    assert extract_keywords(text, top_n=3) == ["search", "vector", "embeddings"]
