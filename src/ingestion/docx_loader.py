"""DOCX loader (python-docx). Headings become section boundaries; tables are flattened."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ingestion.base import BaseLoader, Document, DocumentLoadError, Segment


class DocxLoader(BaseLoader):
    extensions = (".docx",)
    file_type = "docx"

    def _load(self, path: Path) -> Document:
        try:
            from docx import Document as DocxDocument
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:  # pragma: no cover
            raise DocumentLoadError(path, "missing_dependency", "python-docx is not installed.") from exc

        try:
            doc = DocxDocument(str(path))
            segments: list[Segment] = []
            buffer: list[str] = []
            section: Optional[str] = None

            def flush() -> None:
                if buffer:
                    segments.append(Segment(text="\n\n".join(buffer), section=section))
                    buffer.clear()

            for child in doc.element.body.iterchildren():
                if child.tag.endswith("}p"):
                    paragraph = Paragraph(child, doc)
                    text = paragraph.text.strip()
                    if not text:
                        continue
                    style = (paragraph.style.name or "") if paragraph.style is not None else ""
                    if style.lower().startswith("heading") or style.lower() == "title":
                        flush()
                        section = text
                    buffer.append(text)
                elif child.tag.endswith("}tbl"):
                    table = Table(child, doc)
                    for row in table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            buffer.append(" | ".join(cells))
            flush()
            title = doc.core_properties.title or None
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(path, "corrupt_docx", f"Could not parse DOCX: {exc}") from exc

        return self._build(path, segments, {"title": title, "num_sections": len(segments)})
