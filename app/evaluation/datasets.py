"""Versioned evaluation datasets stored as JSONL.

Each line is one question object::

    {"id": "q1",
     "question": "How does the deployment pipeline work?",
     "expected_documents": ["deployment.md"],
     "expected_keywords": ["build", "test"],
     "reference_answer": "Short human-written gold answer."}

``expected_documents`` matches document *names* (stable across re-ingestion),
``expected_keywords`` are matched case-insensitively against retrieved
context, and ``reference_answer`` is an optional gold answer for answer
relevance checks.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_documents: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    reference_answer: str = ""


def load_dataset(path: str | Path) -> list[EvalQuestion]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    questions: list[EvalQuestion] = []
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{dataset_path}:{line_number}: invalid JSON ({exc})") from exc
        if not record.get("question"):
            raise ValueError(f"{dataset_path}:{line_number}: missing 'question'")
        questions.append(
            EvalQuestion(
                id=str(record.get("id") or f"q{line_number}"),
                question=record["question"],
                expected_documents=list(record.get("expected_documents") or []),
                expected_keywords=[str(k).lower() for k in record.get("expected_keywords") or []],
                reference_answer=str(record.get("reference_answer") or ""),
            )
        )
    return questions
