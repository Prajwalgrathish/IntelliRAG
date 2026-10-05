"""Query/document preprocessing helpers and keyword extraction.

Stop words are removed ONLY for lexical utilities (keywords, reranking, evaluation) and
never for text that is fed to a transformer.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.nlp.cleaning import normalize_unicode, remove_control_chars

MAX_QUERY_CHARS = 1000
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


def preprocess_query(query: str, max_chars: int = MAX_QUERY_CHARS) -> str:
    """Normalise a user query: unicode fix, control-char removal, whitespace collapse.

    Raises:
        ValueError: if the query is empty after normalisation.
    """
    if query is None:
        raise ValueError("Query must not be empty.")
    text = remove_control_chars(normalize_unicode(query))
    text = " ".join(text.split())
    if not text:
        raise ValueError("Query must not be empty.")
    return text[:max_chars]


def content_tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens without stop words (lexical utilities only)."""
    tokens = (token.strip("-'").lower() for token in _TOKEN_RE.findall(text))
    return [t for t in tokens if len(t) >= 2 and t not in ENGLISH_STOP_WORDS]


def extract_keywords(text: str, top_n: Optional[int] = 8) -> list[str]:
    """Return the most frequent content tokens (ties broken by first occurrence)."""
    tokens = content_tokens(text)
    counts = Counter(tokens)
    first_seen = {}
    for idx, token in enumerate(tokens):
        first_seen.setdefault(token, idx)
    ranked = sorted(counts, key=lambda t: (-counts[t], first_seen[t]))
    return ranked if top_n is None else ranked[:top_n]
