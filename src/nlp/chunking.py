"""Sentence-aware chunking with word-based size and overlap."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable, Optional

from src.ingestion.base import Segment
from src.nlp.preprocessing import extract_keywords

_ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "dr.", "mr.", "mrs.", "ms.", "prof.", "vs.", "etc.", "fig.", "no.", "inc.", "approx.", "st.", "jr.", "sr.", "al."}
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass
class Chunk:
    """A retrievable unit of text together with its provenance metadata."""

    chunk_id: str
    document: str
    text: str
    source_path: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    word_count: int = 0
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that respects paragraphs and common abbreviations."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        merged: list[str] = []
        for part in _SENTENCE_BOUNDARY_RE.split(paragraph):
            if merged and merged[-1].split()[-1].lower() in _ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        sentences.extend(p.strip() for p in merged if p.strip())
    return sentences


class SentenceChunker:
    """Pack whole sentences into chunks of at most ``chunk_size`` words.

    Consecutive chunks share up to ``chunk_overlap`` words (whole trailing sentences) so that
    information near a boundary is retrievable from either side. Sentences longer than
    ``chunk_size`` are split into word windows.
    """

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_long(self, words: list[str]) -> list[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [" ".join(words[i : i + self.chunk_size]) for i in range(0, len(words), step)]

    def _overlap_tail(self, sentences: list[str]) -> tuple[list[str], int]:
        tail: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            n = len(sentence.split())
            if total + n > self.chunk_overlap:
                break
            tail.insert(0, sentence)
            total += n
        return tail, total

    def chunk_text(self, text: str) -> list[str]:
        """Split ``text`` into overlapping, sentence-aligned chunk strings."""
        units: list[str] = []
        for sentence in split_sentences(text):
            words = sentence.split()
            units.extend(self._split_long(words) if len(words) > self.chunk_size else [sentence])

        chunks: list[str] = []
        current: list[str] = []
        count = 0
        for unit in units:
            n = len(unit.split())
            if current and count + n > self.chunk_size:
                chunks.append(" ".join(current))
                current, count = self._overlap_tail(current)
                if count + n > self.chunk_size:
                    current, count = [], 0
            current.append(unit)
            count += n
        if current:
            chunks.append(" ".join(current))
        return chunks

    def chunk_document(
        self, *, doc_id: str, document_name: str, source_path: str, segments: Iterable[Segment]
    ) -> list[Chunk]:
        """Chunk every segment (page/section) separately so page metadata stays accurate."""
        chunks: list[Chunk] = []
        index = 0
        for segment in segments:
            for text in self.chunk_text(segment.text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{index:04d}",
                        document=document_name,
                        text=text,
                        source_path=source_path,
                        page=segment.page,
                        section=segment.section,
                        chunk_index=index,
                        word_count=len(text.split()),
                        keywords=extract_keywords(text),
                    )
                )
                index += 1
        return chunks
