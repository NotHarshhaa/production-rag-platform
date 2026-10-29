"""Query decomposition for agentic retrieval.

Complex, compound questions ("How does X work and how does Y handle Z?")
retrieve poorly as a single query. A decomposer splits them into focused
sub-queries; the retriever runs each one and fuses the result lists.

Two strategies:
- LLM-based when a real provider is configured (not the deterministic stub).
- A heuristic fallback: split on sentence and conjunction boundaries, then
  keep the halves that are long enough to stand alone as searches.

The original question is always kept as the first sub-query so the fused
result is never worse than plain single-query retrieval.
"""

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONJUNCTION_SPLIT = re.compile(
    r",?\s+(?:and then|and also|also|plus|additionally|in addition)\s+", re.IGNORECASE
)
_MIN_SUBQUERY_WORDS = 3


def _is_stub(llm) -> bool:
    return llm is None or getattr(llm, "name", "base") in ("stub", "base")


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def _heuristic_decompose(question: str) -> list[str]:
    subqueries: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(question):
        sentence = sentence.strip(" .?!")
        if not sentence:
            continue
        parts = _CONJUNCTION_SPLIT.split(sentence)
        for part in parts:
            part = part.strip(" ,.?!")
            if len(_words(part)) >= _MIN_SUBQUERY_WORDS and part.lower() not in (
                s.lower() for s in subqueries
            ):
                subqueries.append(part)
    return subqueries


_LLM_DECOMPOSE_PROMPT = (
    "Break the question below into focused, self-contained search queries.\n"
    "Rules: one query per line, no numbering, no explanations, at most "
    "{max_subqueries} queries, each query must be answerable on its own.\n\n"
    "Question: {question}"
)


def _llm_decompose(question: str, llm, max_subqueries: int) -> list[str]:
    try:
        response = llm.generate(
            [
                {
                    "role": "user",
                    "content": _LLM_DECOMPOSE_PROMPT.format(
                        max_subqueries=max_subqueries, question=question
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:  # decomposition must never break retrieval
        logger.warning("LLM query decomposition failed; using heuristic", extra={"error": str(exc)})
        return []

    subqueries: list[str] = []
    for line in response.text.splitlines():
        line = line.strip().strip("-*0123456789. ")
        if line and len(_words(line)) >= _MIN_SUBQUERY_WORDS:
            subqueries.append(line)
    return subqueries


def decompose_question(
    question: str, llm=None, max_subqueries: int = 3
) -> list[str]:
    """Return the search queries to run for ``question`` (never empty).

    The original question is always first; decomposed sub-queries follow,
    capped at ``max_subqueries`` entries overall. When decomposition yields
    nothing useful, only the original question is returned.
    """

    max_subqueries = max(1, max_subqueries)
    subqueries: list[str] = [question]
    if not _is_stub(llm):
        for candidate in _llm_decompose(question, llm, max_subqueries):
            if candidate.lower() != question.strip().lower():
                subqueries.append(candidate)
    if len(subqueries) < max_subqueries:
        for candidate in _heuristic_decompose(question):
            if len(subqueries) >= max_subqueries:
                break
            if candidate.lower() not in (s.lower() for s in subqueries):
                subqueries.append(candidate)
    return subqueries[:max_subqueries]
