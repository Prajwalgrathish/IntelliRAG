"""Context construction and citation parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.retrieval.types import RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class ContextResult:
    text: str
    used: list[RetrievedChunk] = field(default_factory=list)
    truncated: bool = False
    word_count: int = 0


def format_header(index: int, rc: RetrievedChunk) -> str:
    parts = [f"document: {rc.chunk.document}"]
    if rc.chunk.page is not None:
        parts.append(f"page: {rc.chunk.page}")
    if rc.chunk.section:
        parts.append(f"section: {rc.chunk.section}")
    return f"[{index}] ({', '.join(parts)})"


def build_context(chunks: list[RetrievedChunk], max_words: int = 1500) -> ContextResult:
    """Number retrieved chunks ``[1]..[n]`` and pack them within a word budget.

    The first chunk is always included (truncated if necessary). Later chunks that would
    exceed the budget are omitted, and ``truncated`` is set.
    """
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0
    truncated = False
    for rc in chunks:
        words = rc.chunk.text.split()
        if used and total + len(words) > max_words:
            truncated = True
            break
        if not used and len(words) > max_words:
            words, truncated = words[:max_words], True
        blocks.append(f"{format_header(len(used) + 1, rc)}\n{' '.join(words)}")
        used.append(rc)
        total += len(words)
    return ContextResult(text="\n\n".join(blocks), used=used, truncated=truncated, word_count=total)


def parse_citations(text: str) -> list[int]:
    """Extract citation numbers such as ``[1]`` or ``[2, 3]`` in order of appearance."""
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(text):
        numbers.extend(int(n) for n in re.split(r"\s*,\s*", match.group(1)))
    return numbers


def strip_citations(text: str) -> str:
    return " ".join(_CITATION_RE.sub("", text).split())
