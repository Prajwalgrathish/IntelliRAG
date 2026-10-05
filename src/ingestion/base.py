"""Core ingestion types: documents, segments, errors and the loader interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union


@dataclass
class Segment:
    """A contiguous piece of extracted text (a PDF page or a DOCX section)."""

    text: str
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class Document:
    """A loaded document with extracted text and metadata."""

    name: str
    source_path: str
    file_type: str
    segments: list[Segment]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)


class DocumentLoadError(Exception):
    """Structured, user-presentable document loading failure."""

    def __init__(self, path: Union[str, Path], code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason} ({path})")
        self.path = str(path)
        self.code = code
        self.reason = reason

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "reason": self.reason}


class BaseLoader(ABC):
    """Interface implemented by every format-specific loader."""

    extensions: ClassVar[tuple[str, ...]] = ()
    file_type: ClassVar[str] = ""

    def load(self, path: Union[str, Path]) -> Document:
        """Validate the file and delegate to the format-specific loader."""
        path = Path(path)
        if not path.exists():
            raise DocumentLoadError(path, "file_not_found", "File does not exist.")
        if not path.is_file():
            raise DocumentLoadError(path, "not_a_file", "Path is not a regular file.")
        if path.stat().st_size == 0:
            raise DocumentLoadError(path, "empty_file", "File is empty (0 bytes).")
        return self._load(path)

    @abstractmethod
    def _load(self, path: Path) -> Document:
        """Extract text from a validated file."""

    def _build(
        self, path: Path, segments: list[Segment], metadata: dict[str, Any], empty_hint: str = ""
    ) -> Document:
        segments = [s for s in segments if s.text and s.text.strip()]
        if not segments:
            raise DocumentLoadError(
                path, "no_extractable_text", f"No text could be extracted. {empty_hint}".strip()
            )
        return Document(
            name=path.name,
            source_path=str(path),
            file_type=self.file_type,
            segments=segments,
            metadata=metadata,
        )
