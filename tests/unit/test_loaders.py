"""Unit tests for document loaders."""

import io

import pytest

from app.ingestion.loaders import (
    UnsupportedFileTypeError,
    detect_content_type,
    load_document,
)


def test_detect_content_type_supported():
    assert detect_content_type("a.pdf") == "application/pdf"
    assert detect_content_type("b.TXT") == "text/plain"
    assert detect_content_type("c.md") == "text/markdown"
    assert detect_content_type("d.markdown") == "text/markdown"


def test_detect_content_type_unsupported():
    with pytest.raises(UnsupportedFileTypeError):
        detect_content_type("virus.exe")
    with pytest.raises(UnsupportedFileTypeError):
        detect_content_type("")


def test_load_text_document():
    loaded = load_document("notes.txt", b"hello world")
    assert loaded.text == "hello world"
    assert loaded.content_type == "text/plain"
    assert loaded.pages == ["hello world"]


def test_load_markdown_document():
    loaded = load_document("notes.md", b"# Title\n\nBody")
    assert loaded.content_type == "text/markdown"
    assert "# Title" in loaded.text


def test_load_pdf_document():
    fitz = pytest.importorskip("fitz")

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hello from a PDF page.")
    buffer = io.BytesIO()
    pdf.save(buffer)
    pdf.close()

    loaded = load_document("doc.pdf", buffer.getvalue())
    assert loaded.content_type == "application/pdf"
    assert len(loaded.pages) == 1
    assert "Hello from a PDF page." in loaded.text


def test_load_unsupported_raises():
    with pytest.raises(UnsupportedFileTypeError):
        load_document("thing.docx", b"whatever")
