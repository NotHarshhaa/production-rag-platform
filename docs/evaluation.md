# Evaluation

The platform is evaluated at two layers: **retrieval** (did we fetch the
right evidence?) and **generation** (did we answer well from that
evidence?). This document explains the dataset format, the metrics, how to
run them, and how to use baselines.

## Dataset format

Datasets are JSONL files — one question per line:

```json
{
  "id": "q1",
  "question": "How does the deployment pipeline validate changes?",
  "expected_documents": ["deployment.md"],
  "expected_keywords": ["tests", "ci"],
  "reference_answer": "The CI pipeline runs unit and integration tests on every pull request."
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | no | Stable identifier used in reports (defaults to line number) |
| `question` | yes | The query to run |
| `expected_documents` | no | Document *names* that count as relevant (stable across re-ingestion) |
| `expected_keywords` | no | Case-insensitive keywords that must appear in retrieved context |
| `reference_answer` | no | Gold answer used for answer-relevance scoring |

Keep datasets versioned (`evals/*.jsonl`) and reviewed by hand. The bundled
`evals/sample_questions.jsonl` pairs with the `samples/` corpus.

## Metrics

### Retrieval (exact)

| Metric | Definition |
|---|---|
| Recall@K | Fraction of expected documents found in the top K results |
| Precision@K | Fraction of the top K results that are expected documents |
| MRR | 1 / rank of the first relevant document |

### Context coverage

`keyword_coverage` — fraction of `expected_keywords` that appear anywhere in
the retrieved chunk texts. Catches cases where the "right" document was
retrieved but the wrong part of it.

### Answers (heuristics)

Answer metrics are transparent lexical heuristics, not LLM judges:

| Metric | Definition |
|---|---|
| `faithfulness` | Fraction of answer sentences with lexical support in the retrieved context |
| `answer_relevance` | Token overlap between the answer and the reference answer |
| `citation_precision` | Fraction of citations pointing at chunks actually retrieved |

Treat them as regression signals. A score of 1.0 does not prove the answer
is correct; a drop against baseline does tell you something changed.
Integrating an LLM-as-judge (e.g. RAGAS) is a natural extension behind the
same runner interface.

## Running

```bash
# 1. Ingest the sample corpus (or your own documents)
python scripts/ingest_documents.py samples/

# 2. Retrieval-only evaluation (no LLM calls)
python -m app.evaluation.runner

# 3. Full evaluation including generated answers
python -m app.evaluation.runner --with-answers --top-k 5

# Options
python -m app.evaluation.runner --dataset evals/sample_questions.jsonl \
    --limit 3 --output-dir evals/reports
```

Reports are written to `evals/reports/report-<timestamp>.json` with
per-question records plus aggregate summaries, and a summary is printed to
the console.

## Baseline workflow

1. Pick a dataset and freeze it (never edit a published dataset in place;
   add a new version).
2. Record a baseline report before touching chunking, embeddings, retrieval,
   or prompts (`--with-answers` recommended).
3. Make the change; re-run with identical flags.
4. Compare summaries. Watch for trade-offs (higher recall but lower
   precision, better answers but more tokens) and review per-question
   records for regressions.
5. Note the config block embedded in every report — it captures the
   embedding provider, chunking, and retrieval mode used, so old reports
   stay interpretable.

The evaluation smoke test in CI (`evaluation-smoke` job) ingests `samples/`
and runs the full runner, catching wiring breakage early.
