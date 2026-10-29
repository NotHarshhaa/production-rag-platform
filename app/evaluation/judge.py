"""LLM-as-judge answer grading for the evaluation runner.

A judge LLM grades each generated answer on three 0-5 rubric scales —
faithfulness (claims supported by the context), relevance (answers the
question / matches the reference), and citation support (cited passages
contain what the answer used). Scores are normalized to 0-1.

The deterministic stub LLM cannot follow the rubric, so when the configured
provider is the stub (or the judge output is unparseable) the grader falls
back to the transparent lexical heuristics from app.evaluation.metrics and
marks the result ``judge: "heuristic"`` so reports stay honest about how
each score was produced.
"""

import json
import re

from app.core.logging import get_logger
from app.evaluation.metrics import answer_relevance_estimate, faithfulness_estimate

logger = get_logger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_PROMPT = """\
You are a strict evaluator of Retrieval-Augmented Generation answers.

Score the candidate answer on three integer scales from 0 to 5:
- "faithfulness": every claim in the answer is supported by the context passages.
- "relevance": the answer addresses the question{reference_clause}.
- "citation_support": the passages the answer cites actually contain the information used.

Respond with ONLY a JSON object, no other text:
{{"faithfulness": <0-5>, "relevance": <0-5>, "citation_support": <0-5>}}

Question: {question}
{reference_line}Context passages:
{context}

Candidate answer: {answer}"""


def _build_judge_messages(
    question: str, answer: str, context_texts: list[str], reference_answer: str | None
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{position}] {text}" for position, text in enumerate(context_texts, start=1)
    )
    reference_clause = " and matches the reference answer" if reference_answer else ""
    reference_line = f"Reference answer: {reference_answer}\n" if reference_answer else ""
    prompt = JUDGE_PROMPT.format(
        reference_clause=reference_clause,
        question=question,
        reference_line=reference_line,
        context=context,
        answer=answer,
    )
    return [{"role": "user", "content": prompt}]


def _parse_scores(text: str) -> dict[str, float] | None:
    """Extract the three rubric scores from the judge output, or None."""

    match = _JSON_OBJECT.search(text)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    scores: dict[str, float] = {}
    for key in ("faithfulness", "relevance", "citation_support"):
        value = raw.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        scores[key] = max(0.0, min(5.0, float(value))) / 5.0
    return scores


def _heuristic_scores(
    answer: str, context_texts: list[str], question: str, reference_answer: str | None
) -> dict[str, float]:
    # With no context (insufficient-context answers) every heuristic is 0.
    faithfulness = faithfulness_estimate(answer, context_texts) if context_texts else 0.0
    return {
        "faithfulness": faithfulness,
        "relevance": answer_relevance_estimate(answer, reference_answer or question),
        "citation_support": faithfulness,
    }


def _result(model: str, judge: str, scores: dict[str, float]) -> dict:
    return {
        "faithfulness": round(scores["faithfulness"], 4),
        "relevance": round(scores["relevance"], 4),
        "citation_support": round(scores["citation_support"], 4),
        "model": model,
        "judge": judge,
    }


def judge_answer(
    llm,
    question: str,
    answer: str,
    context_texts: list[str],
    reference_answer: str | None = None,
) -> dict:
    """Grade one answer; returns normalized 0-1 scores plus judge provenance."""

    def fallback() -> dict:
        return _result(
            "heuristic-fallback",
            "heuristic",
            _heuristic_scores(answer, context_texts, question, reference_answer),
        )

    if getattr(llm, "name", "base") in ("stub", "base"):
        return fallback()

    try:
        response = llm.generate(
            _build_judge_messages(question, answer, context_texts, reference_answer),
            temperature=0.0,
            max_tokens=128,
        )
    except Exception as exc:  # a judge failure must never fail the evaluation
        logger.warning("Judge LLM failed; falling back to heuristics", extra={"error": str(exc)})
        return fallback()

    scores = _parse_scores(response.text)
    if scores is None:
        logger.warning("Judge output was not valid JSON; falling back to heuristics")
        return fallback()

    return _result(response.model or llm.model, "llm", scores)
