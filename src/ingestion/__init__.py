"""Document loaders and loader registry."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from src.ingestion.base import BaseLoader, Document, DocumentLoadError, Segment
from src.ingestion.docx_loader import DocxLoader
from src.ingestion.pdf_loader import PdfLoader
from src.ingestion.txt_loader import TxtLoader

_LOADERS: tuple[BaseLoader, ...] = (TxtLoader(), PdfLoader(), DocxLoader())
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted({e for l in _LOADERS for e in l.extensions}))


def get_loader(path: Union[str, Path]) -> BaseLoader:
    """Return the loader for a file extension or raise a structured error."""
    suffix = Path(path).suffix.lower()
    for loader in _LOADERS:
        if suffix in loader.extensions:
            return loader
    raise DocumentLoadError(
        path, "unsupported_type", f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def load_document(path: Union[str, Path]) -> Document:
    """Load a document with the loader that matches its extension."""
    return get_loader(path).load(path)


__all__ = [
    "BaseLoader", "Document", "DocumentLoadError", "Segment",
    "SUPPORTED_EXTENSIONS", "get_loader", "load_document",
]
