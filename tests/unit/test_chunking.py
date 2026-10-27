"""Unit tests for chunking."""

import pytest

from app.ingestion.chunking import chunk_document, chunk_text, estimate_tokens, tokenize
from app.ingestion.loaders import LoadedDocument


def _paragraph(seed: str, sentences: int = 5) -> str:
    return " ".join(f"Sentence {seed} number {i} about testing chunks." for i in range(sentences))


def test_short_text_produces_single_chunk():
    chunks = chunk_text("Hello world, this is short.", page=1, size=800, overlap=120)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world, this is short."
    assert chunks[0].page == 1
    assert chunks[0].index == 0


def test_empty_text_produces_no_chunks():
    assert chunk_text("", page=1, size=800, overlap=120) == []
    assert chunk_text("   \n\n  \n", page=1, size=800, overlap=120) == []


def test_long_text_respects_size_limit():
    text = "\n\n".join(_paragraph(f"p{i}") for i in range(10))
    chunks = chunk_text(text, page=1, size=400, overlap=60)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 400


def test_consecutive_chunks_overlap_in_long_runs():
    single_paragraph = _paragraph("dense", sentences=40)  # one long paragraph
    chunks = chunk_text(single_paragraph, page=1, size=300, overlap=80)
    assert len(chunks) > 1
    # The sliding window guarantees the shared region: the tail of the
    # previous chunk appears in the next one.
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.text[-30:] in current.text


def test_tail_carry_across_paragraph_chunks():
    paragraphs = [_paragraph(f"p{i}", sentences=3) for i in range(8)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, page=1, size=400, overlap=60)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.text[-30:] in current.text


def test_indices_are_contiguous():
    text = "\n\n".join(_paragraph(f"p{i}") for i in range(6))
    chunks = chunk_text(text, page=1, size=400, overlap=60, start_index=7)
    assert [c.index for c in chunks] == list(range(7, 7 + len(chunks)))


def test_invalid_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", page=1, size=0, overlap=0)


def test_overlap_larger_than_size_is_clamped():
    chunks = chunk_text("word " * 200, page=1, size=100, overlap=500)
    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_chunk_document_tracks_pages():
    loaded = LoadedDocument(
        name="doc.md",
        content_type="text/markdown",
        text="page1",
        pages=["Page one content. " * 20, "Page two content. " * 20, ""],
    )
    chunks = chunk_document(loaded, size=200, overlap=20)
    assert chunks
    pages_used = {c.page for c in chunks}
    assert pages_used <= {1, 2}
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.char_count == len(chunk.text)
        assert chunk.token_count > 0


def test_estimate_tokens_and_tokenize():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]
