"""LLM-as-judge grading and the --judge evaluation runner flag."""

from app.evaluation.judge import _parse_scores, judge_answer
from app.generation.answer import LLMResponse
from tests.integration.conftest import DOC_ONE, DOC_TWO


class FakeLLM:
    name = "openai"
    model = "fake-judge"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, messages, temperature, max_tokens) -> LLMResponse:
        return LLMResponse(text=self.text, prompt_tokens=10, completion_tokens=5, model=self.model)


def test_parse_scores_normalizes_and_clamps():
    assert _parse_scores('{"faithfulness": 5, "relevance": 0, "citation_support": 3}') == {
        "faithfulness": 1.0,
        "relevance": 0.0,
        "citation_support": 0.6,
    }
    assert _parse_scores('{"faithfulness": 9, "relevance": -2, "citation_support": 2.5}') == {
        "faithfulness": 1.0,
        "relevance": 0.0,
        "citation_support": 0.5,
    }


def test_parse_scores_rejects_malformed_output():
    assert _parse_scores("no json here") is None
    assert _parse_scores('{"faithfulness": "high", "relevance": 3, "citation_support": 3}') is None
    assert _parse_scores('{"faithfulness": true, "relevance": 3, "citation_support": 3}') is None
    assert _parse_scores('{"faithfulness": 3, "relevance": 3}') is None  # missing key


def test_judge_with_llm_scores(tmp_path):
    llm = FakeLLM('Verdict: {"faithfulness": 4, "relevance": 5, "citation_support": 2}')
    result = judge_answer(llm, "q?", "answer [1]", ["context"], "reference")
    assert result == {
        "faithfulness": 0.8,
        "relevance": 1.0,
        "citation_support": 0.4,
        "model": "fake-judge",
        "judge": "llm",
    }


def test_judge_falls_back_when_output_is_not_json(tmp_path):
    result = judge_answer(FakeLLM("I cannot grade this."), "q?", "answer", ["context"])
    assert result["model"] == "heuristic-fallback"
    assert result["judge"] == "heuristic"


def test_judge_with_stub_uses_heuristics(tmp_path):
    from app.generation.answer import StubLLM

    result = judge_answer(StubLLM(), "q?", "answer", ["context"])
    assert result["judge"] == "heuristic"
    assert set(result) == {"faithfulness", "relevance", "citation_support", "model", "judge"}


def test_runner_with_judge_produces_judge_section(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'eval.db'}")
    monkeypatch.setenv("VECTOR_STORE", "inmemory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "64")
    monkeypatch.setenv("LLM_PROVIDER", "stub")

    from app.api.dependencies import build_container
    from app.core.config import get_settings
    from app.evaluation.runner import DEFAULT_DATASET, run_evaluation

    get_settings.cache_clear()
    try:
        container = build_container(get_settings())
        for name, doc in (("deployment.md", DOC_ONE), ("security.md", DOC_TWO)):
            document_id, _ = container.pipeline.ingest(name=name, data=doc.encode("utf-8"))
            container.pipeline.index_document(document_id)

        report = run_evaluation(
            dataset_path=DEFAULT_DATASET, top_k=3, with_judge=True, container=container
        )
        assert report["summary"]["with_answers"] is True
        judge_summary = report["summary"]["judge"]
        assert judge_summary["model"] == "heuristic-fallback"
        assert 0.0 <= judge_summary["faithfulness"] <= 1.0
        per_question = [r["judge"] for r in report["results"] if "judge" in r]
        assert per_question, "every answered question should carry judge scores"
        assert report["config"]["judge_model"] == "stub-extractive"
    finally:
        get_settings.cache_clear()
