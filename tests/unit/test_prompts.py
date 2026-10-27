"""Unit tests for prompt construction."""

from app.generation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_context, build_messages
from app.retrieval.hybrid_search import RetrievedChunk


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc:{i}",
        document_id="doc",
        document_name="doc.md",
        page=i,
        text=f"Chunk text {i}. " * 10,
        score=0.9,
    )


def test_build_context_numbers_passages():
    context = build_context([_chunk(1), _chunk(2)])
    assert "[1] doc.md — page 1" in context
    assert "[2] doc.md — page 2" in context
    assert "Chunk text 1." in context


def test_build_context_truncates_long_chunks():
    chunk = _chunk(1)
    chunk.text = "word " * 1000
    context = build_context([chunk], max_chars_per_chunk=50)
    assert len(context) < 200
    assert context.rstrip().endswith("...")


def test_build_messages_structure():
    messages = build_messages("What is RAG?", "context here")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "context here" in messages[1]["content"]
    assert "What is RAG?" in messages[1]["content"]
    assert "Ignore" not in messages[1]["content"]


def test_system_prompt_mentions_citations_and_injection_defense():
    assert "[1]" in SYSTEM_PROMPT
    assert "untrusted" in SYSTEM_PROMPT.lower()
    assert PROMPT_VERSION


def test_stub_llm_answers_with_citation():
    from app.generation.answer import StubLLM

    llm = StubLLM()
    messages = build_messages("What does the pipeline run?", build_context([_chunk(1)]))
    response = llm.generate(messages, temperature=0.0, max_tokens=128)
    assert "[1]" in response.text
    assert response.completion_tokens > 0
    assert response.model == "stub-extractive"
