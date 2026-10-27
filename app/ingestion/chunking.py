"""Character-based chunking that prefers paragraph boundaries.

Chunks never span pages, so every citation carries an exact page number.
Long paragraphs that exceed the chunk size are split with a sliding window
so ``overlap`` still applies within dense text runs.
"""

import re
from dataclasses import dataclass, field

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 characters per token)."""

    return max(1, len(text) // 4) if text else 0


@dataclass
class Chunk:
    text: str
    index: int
    page: int
    pages: list[int] = field(default_factory=list)
    char_count: int = 0
    token_count: int = 0


def _split_windows(text: str, size: int, overlap: int) -> list[str]:
    """Split a text longer than ``size`` into overlapping windows."""

    step = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return windows


def _tail(text: str, overlap: int) -> str:
    """Take the trailing ``overlap`` characters, snapped to a word boundary."""

    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    if space != -1 and len(tail) > 1:
        tail = tail[space + 1 :]
    return tail.strip()


def chunk_text(text: str, page: int, size: int, overlap: int, start_index: int = 0) -> list[Chunk]:
    """Chunk a single page of text into pieces of at most ``size`` characters.

    Paragraph boundaries are preferred; a long paragraph is split with a
    sliding window so ``overlap`` still applies within dense text runs, and
    a word-aligned tail of each chunk is carried into the next one.
    """

    if size < 1:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size - 1))

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    # Pre-split oversized paragraphs into overlapping windows.
    blocks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            blocks.append(paragraph)
        else:
            blocks.extend(_split_windows(paragraph, size, overlap))

    # Greedily pack blocks into chunks up to the size budget, carrying a
    # word-aligned tail of the previous chunk into the next one.
    packed: list[str] = []
    current = ""
    prefix = ""
    for block in blocks:
        if current:
            candidate = f"{current}\n\n{block}"
        else:
            candidate = f"{prefix}\n\n{block}" if prefix else block
            if len(candidate) > size:
                candidate = block  # the tail does not fit; start clean
        if len(candidate) > size:
            if current:
                packed.append(current)
                prefix = _tail(current, overlap) if overlap > 0 else ""
                candidate = f"{prefix}\n\n{block}" if prefix else block
                if len(candidate) > size:
                    candidate = block
            else:
                candidate = block[:size]  # defensive; blocks are pre-split
        current = candidate
    if current:
        packed.append(current)

    return [
        Chunk(
            text=chunk,
            index=start_index + position,
            page=page,
            pages=[page],
            char_count=len(chunk),
            token_count=estimate_tokens(chunk),
        )
        for position, chunk in enumerate(packed)
    ]


def chunk_document(loaded, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Chunk a LoadedDocument page by page with a continuous chunk index."""

    chunks: list[Chunk] = []
    next_index = 0
    for page_number, page_text in enumerate(loaded.pages, start=1):
        page_chunks = chunk_text(page_text, page_number, size, overlap, next_index)
        chunks.extend(page_chunks)
        next_index += len(page_chunks)
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens shared by BM25 and the hashing embedder."""

    return _WORD_PATTERN.findall(text.lower())
