"""Evaluation metrics for retrieval and answer quality.

Retrieval metrics are exact (recall@k, precision@k, MRR). Answer metrics are
transparent lexical heuristics — useful regression signals, not proof of
correctness; validate important results against human review (see
docs/evaluation.md).
"""

import re

from app.ingestion.chunking import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _hits_at_k(retrieved: list[str], relevant: set[str], k: int) -> list[str]:
    return [item for item in retrieved[:k] if item in relevant]


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""

    if k <= 0 or not retrieved[:k]:
        return 0.0
    relevant_set = set(relevant)
    hits = _hits_at_k(retrieved, relevant_set, k)
    return len(hits) / min(k, len(retrieved))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant items found within the top-k."""

    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = _hits_at_k(retrieved, relevant_set, k)
    return len(hits) / len(relevant_set)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """Mean reciprocal rank of the first relevant item (single question)."""

    relevant_set = set(relevant)
    for position, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def keyword_coverage(texts: list[str], keywords: list[str]) -> float:
    """Fraction of keywords that appear in the given texts (case-insensitive)."""

    if not keywords:
        return 0.0
    haystack = " ".join(texts).lower()
    found = sum(1 for keyword in keywords if keyword.lower() in haystack)
    return found / len(keywords)


def faithfulness_estimate(answer: str, context_texts: list[str]) -> float:
    """Heuristic faithfulness: fraction of answer sentences with lexical
    support in the retrieved context.

    A sentence counts as supported when at least half of its non-stopword
    tokens (min 1) appear somewhere in the context. Sentences that are pure
    citations or short connectives are ignored.
    """

    context_tokens = set()
    for text in context_texts:
        context_tokens.update(tokenize(text))

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
    scored = 0
    supported = 0
    for sentence in sentences:
        tokens = [t for t in tokenize(sentence) if len(t) > 2]
        if not tokens:
            continue
        scored += 1
        support = sum(1 for token in tokens if token in context_tokens)
        if support >= max(1, len(tokens) // 2):
            supported += 1
    return supported / scored if scored else 0.0


def answer_relevance_estimate(answer: str, question: str) -> float:
    """Heuristic answer relevance: overlap between question and answer tokens."""

    question_tokens = {t for t in tokenize(question) if len(t) > 2}
    answer_tokens = {t for t in tokenize(answer) if len(t) > 2}
    if not question_tokens:
        return 0.0
    overlap = question_tokens & answer_tokens
    return len(overlap) / len(question_tokens)


def citation_precision(citation_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> float:
    """Fraction of citations that point at chunks actually retrieved."""

    if not citation_chunk_ids:
        return 0.0
    retrieved = set(retrieved_chunk_ids)
    valid = sum(1 for chunk_id in citation_chunk_ids if chunk_id in retrieved)
    return valid / len(citation_chunk_ids)


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
