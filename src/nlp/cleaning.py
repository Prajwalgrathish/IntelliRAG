"""Text cleaning that is safe for transformer models.

Deliberately NOT done here: lower-casing, stop-word removal, stemming/lemmatisation or
punctuation stripping. Transformer tokenizers were trained on natural text and these
steps would remove semantic signal (negation, word order, casing of named entities).
"""
from __future__ import annotations

import re
import unicodedata

_QUOTE_MAP = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2022": "-"}
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]{2,})-\n([a-z]{2,})")
_PAGE_NUMBER_RE = re.compile(r"^[ \t]*(?:page[ \t]+)?\d{1,3}(?:[ \t]*(?:/|of)[ \t]*\d{1,3})?[ \t]*$", re.IGNORECASE | re.MULTILINE)


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalisation (fixes ligatures, full-width forms) and straighten quotes."""
    return unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)


def remove_control_chars(text: str) -> str:
    """Remove control and zero-width characters while keeping newlines and tabs."""
    return _ZERO_WIDTH_RE.sub("", _CONTROL_RE.sub("", text))


def fix_hyphenation(text: str) -> str:
    """Re-join words split across lines by a hyphen (common in PDF extraction)."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def remove_page_artifacts(text: str) -> str:
    """Drop lines that only contain a page number such as ``12`` or ``Page 3 of 10``."""
    return _PAGE_NUMBER_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines, preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Full document-cleaning pipeline used before chunking."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = fix_hyphenation(text)
    text = remove_page_artifacts(text)
    return normalize_whitespace(text)
