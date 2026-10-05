"""Plain-text / Markdown loader."""
from __future__ import annotations

from pathlib import Path

from src.ingestion.base import BaseLoader, Document, DocumentLoadError, Segment


class TxtLoader(BaseLoader):
    extensions = (".txt", ".md")
    file_type = "txt"
    _ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

    def _load(self, path: Path) -> Document:
        raw = path.read_bytes()
        text, used = None, None
        for encoding in self._ENCODINGS:
            try:
                text, used = raw.decode(encoding), encoding
                break
            except UnicodeDecodeError:
                continue
        if text is None:  # pragma: no cover - latin-1 never fails
            raise DocumentLoadError(path, "decode_error", "Could not decode file as text.")
        return self._build(path, [Segment(text=text)], {"encoding": used, "size_bytes": len(raw)})
