"""Ask a question from the command line using the configured retrieval stack.

Usage:
    python scripts/query.py "How does the deployment pipeline work?"
    python scripts/query.py "What is prompt injection?" --top-k 3 --mode semantic

Uses the same settings and components as the API (settings from .env or the
environment). Make sure the corpus is indexed first:
    python scripts/ingest_documents.py samples/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.dependencies import build_container  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the RAG platform from the CLI")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--mode", choices=["hybrid", "semantic"], default=None)
    parser.add_argument(
        "--document-ids", nargs="*", default=None, help="Restrict retrieval to document ids"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    container = build_container(settings)

    metadata_filter = {"document_id": args.document_ids} if args.document_ids else None
    result = container.answer_service.answer(
        question=args.question, top_k=args.top_k, metadata_filter=metadata_filter, mode=args.mode
    )

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")

    if result.insufficient_context:
        print("(no relevant documents found)")
    for citation in result.citations:
        print(
            f"  [{citation.document_name} p.{citation.page}] {citation.chunk_id}"
            f" (score {citation.score})"
        )

    usage = result.usage
    print(
        f"\n({result.retrieval_mode} | {usage.get('total_tokens', 0)} tokens"
        + (f" | ${usage.get('cost_usd', 0):.6f}" if usage.get("cost_usd") else "")
        + (" | cached" if result.cached else "")
        + f" | {result.total_latency_ms:.0f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
