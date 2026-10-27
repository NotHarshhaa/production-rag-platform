"""Evaluation runner smoke test against a freshly ingested sample corpus."""

import json

from app.api.dependencies import build_container
from app.evaluation.runner import run_evaluation
from tests.integration.conftest import make_settings

SAMPLE_DIR = "samples"


def test_run_evaluation_against_sample_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(__import__("pathlib").Path(__file__).resolve().parents[2])
    settings = make_settings(tmp_path)
    container = build_container(settings)

    from scripts.ingest_documents import collect_files

    files = collect_files(__import__("pathlib").Path(SAMPLE_DIR))
    assert files, "sample corpus must exist"

    for file_path in files:
        document_id, duplicate = container.pipeline.ingest(file_path.name, file_path.read_bytes())
        assert not duplicate
        container.pipeline.index_document(document_id)

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps({
            "id": "q1",
            "question": "How does the deployment pipeline validate changes?",
            "expected_documents": ["deployment.md"],
            "expected_keywords": ["tests", "ci"],
            "reference_answer": "The CI pipeline runs tests before merge.",
        })
        + "\n",
        encoding="utf-8",
    )

    report = run_evaluation(
        dataset_path=dataset, top_k=3, with_answers=True, container=container
    )

    assert report["summary"]["questions"] == 1
    retrieval = report["summary"]["retrieval"]
    assert 0.0 <= retrieval["recall_at_k"] <= 1.0
    assert 0.0 <= retrieval["mrr"] <= 1.0
    assert 0.0 <= report["summary"]["context"]["keyword_coverage"] <= 1.0
    answer = report["summary"]["answer"]
    assert 0.0 <= answer["faithfulness"] <= 1.0

    record = report["results"][0]
    assert record["retrieved_documents"]
    assert record["answer"]["insufficient_context"] is False


def test_run_evaluation_empty_report_flags_missing_corpus(tmp_path):
    settings = make_settings(tmp_path)
    container = build_container(settings)
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({"id": "q1", "question": "anything"}) + "\n", encoding="utf-8")

    report = run_evaluation(dataset_path=dataset, top_k=3, container=container)
    assert report["summary"]["questions"] == 1
    assert report["results"][0]["chunks_returned"] == 0
