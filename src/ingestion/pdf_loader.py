"""PDF loader based on pypdf (text-layer extraction only; no OCR)."""
from __future__ import annotations

from pathlib import Path

from src.ingestion.base import BaseLoader, Document, DocumentLoadError, Segment


class PdfLoader(BaseLoader):
    extensions = (".pdf",)
    file_type = "pdf"

    def _load(self, path: Path) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise DocumentLoadError(path, "missing_dependency", "pypdf is not installed.") from exc

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise DocumentLoadError(path, "encrypted_pdf", "PDF is password protected.")
            segments: list[Segment] = []
            empty_pages = 0
            for number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    segments.append(Segment(text=text, page=number))
                else:
                    empty_pages += 1
            info = reader.metadata
            metadata = {
                "num_pages": len(reader.pages),
                "empty_pages": empty_pages,
                "title": getattr(info, "title", None) if info else None,
            }
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(path, "corrupt_pdf", f"Could not parse PDF: {exc}") from exc

        return self._build(
            path, segments, metadata, empty_hint="The PDF may contain scanned images; OCR is not supported."
        )
