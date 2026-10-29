"""Evaluation runner: score retrieval and answers over a JSONL dataset.

Usage:
    python -m app.evaluation.runner --dataset evals/sample_questions.jsonl
    python -m app.evaluation.runner --with-answers --top-k 5
    python -m app.evaluation.runner --with-answers --judge

The runner reuses the configured retrieval stack (see app.api.dependencies).
Reports are written to evals/reports/ and printed as a compact summary.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.api.dependencies import build_container
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.evaluation.datasets import EvalQuestion, load_dataset
from app.evaluation.judge import judge_answer
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
from app.generation.prompts import PROMPT_VERSION

logger = get_logger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evals" / "sample_questions.jsonl"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "evals" / "reports"


def evaluate_question(
    item: EvalQuestion,
    container,
    top_k: int,
    with_answers: bool,
    with_judge: bool = False,
) -> dict:
    started = time.perf_counter()
    if with_answers:
        result = container.answer_service.answer(question=item.question, top_k=top_k)
        chunks = result.chunks
        answer_text = result.answer
        citations = result.citations
        usage = result.usage
        insufficient = result.insufficient_context
    else:
        chunks = container.retriever.retrieve(item.question, top_k=top_k)
        answer_text = ""
        citations = []
        usage = {}
        insufficient = not chunks
    latency_ms = (time.perf_counter() - started) * 1000

    # Doc-level metrics expect unique documents, one entry per chunk hit.
    document_ids = list(dict.fromkeys(chunk.document_name for chunk in chunks))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    context_texts = [chunk.text for chunk in chunks]

    record: dict = {
        "id": item.id,
        "question": item.question,
        "retrieved_documents": document_ids,
        "chunks_returned": len(chunks),
        "latency_ms": round(latency_ms, 2),
    }

    if item.expected_documents:
        record["retrieval"] = {
            "recall_at_k": recall_at_k(document_ids, item.expected_documents, top_k),
            "precision_at_k": precision_at_k(document_ids, item.expected_documents, top_k),
            "mrr": mrr(document_ids, item.expected_documents),
        }
    if item.expected_keywords:
        record["context"] = {
            "keyword_coverage": keyword_coverage(context_texts, item.expected_keywords)
        }

    if with_answers:
        answer_metrics: dict = {
            "insufficient_context": insufficient,
        }
        if item.reference_answer:
            answer_metrics["answer_relevance"] = answer_relevance_estimate(
                answer_text, item.reference_answer
            )
        if context_texts:
            answer_metrics["faithfulness"] = faithfulness_estimate(answer_text, context_texts)
        if citations:
            answer_metrics["citation_precision"] = citation_precision(
                [c.chunk_id for c in citations], chunk_ids
            )
        record["answer"] = answer_metrics
        record["usage"] = usage
        if with_judge:
            record["judge"] = judge_answer(
                container.answer_service.llm,
                question=item.question,
                answer=answer_text,
                context_texts=context_texts,
                reference_answer=item.reference_answer,
            )
    return record


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET,
    top_k: int = 5,
    with_answers: bool = False,
    with_judge: bool = False,
    limit: int | None = None,
    container=None,
) -> dict:
    settings = get_settings()
    if container is None:
        container = build_container(settings)

    with_answers = with_answers or with_judge
    questions = load_dataset(dataset_path)
    if limit:
        questions = questions[:limit]

    records = [
        evaluate_question(item, container, top_k, with_answers, with_judge)
        for item in questions
    ]

    retrieval_recall = [
        r["retrieval"]["recall_at_k"] for r in records if "retrieval" in r
    ]
    retrieval_precision = [
        r["retrieval"]["precision_at_k"] for r in records if "retrieval" in r
    ]
    retrieval_mrr = [r["retrieval"]["mrr"] for r in records if "retrieval" in r]
    context_coverage = [r["context"]["keyword_coverage"] for r in records if "context" in r]

    summary: dict = {
        "questions": len(records),
        "top_k": top_k,
        "with_answers": with_answers,
        "retrieval": {
            "recall_at_k": mean(retrieval_recall),
            "precision_at_k": mean(retrieval_precision),
            "mrr": mean(retrieval_mrr),
        },
        "context": {"keyword_coverage": mean(context_coverage)},
        "latency_ms_p50_p95": _percentiles([r["latency_ms"] for r in records]),
    }
    if with_answers:
        summary["answer"] = {
            "faithfulness": mean(
                [r["answer"]["faithfulness"] for r in records if "faithfulness" in r["answer"]]
            ),
            "answer_relevance": mean(
                [
                    r["answer"]["answer_relevance"]
                    for r in records
                    if "answer_relevance" in r["answer"]
                ]
            ),
            "citation_precision": mean(
                [
                    r["answer"]["citation_precision"]
                    for r in records
                    if "citation_precision" in r["answer"]
                ]
            ),
        }
        summary["tokens"] = {
            "total": sum(r.get("usage", {}).get("total_tokens", 0) for r in records)
        }
    if with_judge:
        judged = [r["judge"] for r in records if "judge" in r]
        summary["judge"] = {
            "model": judged[0]["model"] if judged else "n/a",
            "mode": judged[0]["judge"] if judged else "n/a",
            "faithfulness": mean([j["faithfulness"] for j in judged]),
            "relevance": mean([j["relevance"] for j in judged]),
            "citation_support": mean([j["citation_support"] for j in judged]),
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "config": {
            "embedding_provider": settings.embedding_provider,
            "vector_store": settings.vector_store,
            "hybrid_retrieval": settings.hybrid_retrieval,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "prompt_version": PROMPT_VERSION,
            **(
                {"judge_model": container.answer_service.llm.model} if with_judge else {}
            ),
        },
        "summary": summary,
        "results": records,
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 2)

    return {"p50": percentile(0.50), "p95": percentile(0.95)}


def print_summary(report: dict) -> None:
    summary = report["summary"]
    print("\n=== Evaluation summary ===")
    print(f"Dataset:        {report['dataset']} ({summary['questions']} questions)")
    print(f"Config:         {json.dumps(report['config'], indent=None)}")
    retrieval = summary["retrieval"]
    print(
        f"Retrieval:      recall@k={retrieval['recall_at_k']}  "
        f"precision@k={retrieval['precision_at_k']}  mrr={retrieval['mrr']}"
    )
    print(f"Context:        keyword_coverage={summary['context']['keyword_coverage']}")
    if "answer" in summary:
        answer = summary["answer"]
        print(
            f"Answers:        faithfulness={answer['faithfulness']}  "
            f"relevance={answer['answer_relevance']}  "
            f"citation_precision={answer['citation_precision']}"
        )
        print(f"Tokens:         total={summary['tokens']['total']}")
    if "judge" in summary:
        judge = summary["judge"]
        print(
            f"Judge ({judge['model']}, {judge['mode']}):  "
            f"faithfulness={judge['faithfulness']}  "
            f"relevance={judge['relevance']}  "
            f"citation_support={judge['citation_support']}"
        )
    print(f"Latency ms:     {summary['latency_ms_p50_p95']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation over a JSONL dataset")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to JSONL dataset")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--with-answers", action="store_true", help="Also generate and score answers"
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Grade answers with the configured LLM as judge (implies --with-answers)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    report = run_evaluation(
        dataset_path=args.dataset,
        top_k=args.top_k,
        with_answers=args.with_answers,
        with_judge=args.judge,
        limit=args.limit,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"report-{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report)
    print(f"Report saved:   {report_path}")

    if not report["results"]:
        print("WARNING: no questions evaluated — is the corpus indexed?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
