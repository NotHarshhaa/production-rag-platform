"""Load raw document bytes into normalized text.

PDF is handled with PyMuPDF (page-aware); plain text and Markdown are decoded
as UTF-8 with a single synthetic page. Other file types are rejected early so
unsupported uploads never enter the pipeline.
"""

import hashlib
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


class UnsupportedFileTypeError(ValueError):
    """Raised when a document has an extension we cannot process."""


@dataclass
class LoadedDocument:
    name: str
    content_type: str
    text: str
    pages: list[str] = field(default_factory=list)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def detect_content_type(filename: str) -> str:
    """Map a filename to its supported content type, or raise."""

    name = (filename or "").strip().lower()
    for extension, content_type in SUPPORTED_EXTENSIONS.items():
        if name.endswith(extension):
            return content_type
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise UnsupportedFileTypeError(
        f"Unsupported file type for {filename!r}. Supported types: {supported}"
    )


def load_document(filename: str, data: bytes) -> LoadedDocument:
    """Extract text from raw document bytes, preserving page boundaries."""

    content_type = detect_content_type(filename)
    if content_type == "application/pdf":
        return _load_pdf(filename, data)
    return _load_text(filename, content_type, data)


def _load_pdf(filename: str, data: bytes) -> LoadedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - pymupdf is a core dependency
        raise RuntimeError(
            "PDF support requires PyMuPDF. Install it with: pip install pymupdf"
        ) from exc

    pages: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as pdf:
        for page in pdf:
            pages.append(page.get_text("text"))

    text = "\n\n".join(page for page in pages if page.strip())
    logger.info("Loaded PDF document", extra={"file": filename, "pages": len(pages)})
    return LoadedDocument(
        name=filename,
        content_type="application/pdf",
        text=text,
        pages=pages,
    )


def _load_text(filename: str, content_type: str, data: bytes) -> LoadedDocument:
    text = data.decode("utf-8", errors="replace")
    return LoadedDocument(
        name=filename,
        content_type=content_type,
        text=text,
        pages=[text],
    )
