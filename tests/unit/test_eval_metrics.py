"""Unit tests for evaluation metrics math."""

import pytest

from app.evaluation.metrics import (
    answer_relevance_estimate,
    citation_precision,
    faithfulness_estimate,
    keyword_coverage,
    mean,
    mrr,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["b"], k=2) == 0.5
    assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0
    assert precision_at_k(["x", "y"], ["a"], k=2) == 0.0
    assert precision_at_k([], ["a"], k=2) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
    assert recall_at_k(["a", "b"], ["a", "c"], k=2) == 0.5
    assert recall_at_k(["a"], ["a", "b", "c"], k=3) == pytest.approx(1 / 3)
    assert recall_at_k(["x"], [], k=3) == 0.0


def test_mrr():
    assert mrr(["x", "a", "b"], ["a"]) == 0.5
    assert mrr(["a"], ["a"]) == 1.0
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_keyword_coverage():
    texts = ["Docker containers package applications", "Kubernetes scales them"]
    assert keyword_coverage(texts, ["docker", "kubernetes"]) == 1.0
    assert keyword_coverage(texts, ["docker", "postgres"]) == 0.5
    assert keyword_coverage(texts, []) == 0.0


def test_faithfulness_estimate():
    context = ["The deployment pipeline runs tests before releasing to production."]
    supported = faithfulness_estimate("The pipeline runs tests before release.", context)
    unsupported = faithfulness_estimate("Cats are excellent swimmers in oceans.", context)
    assert supported > 0.5
    assert unsupported < 0.5


def test_answer_relevance_estimate():
    score = answer_relevance_estimate("The pipeline runs tests", "How does the pipeline run?")
    assert 0.0 < score <= 1.0
    assert answer_relevance_estimate("", "question tokens here") == 0.0


def test_citation_precision():
    assert citation_precision(["c1", "c2"], ["c1", "c3"]) == 0.5
    assert citation_precision([], ["c1"]) == 0.0


def test_mean():
    assert mean([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert mean([]) == 0.0
