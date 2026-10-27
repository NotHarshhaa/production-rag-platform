"""Ingest documents into the RAG platform from the command line.

Usage:
    python scripts/ingest_documents.py samples/
    python scripts/ingest_documents.py path/to/report.pdf --no-index
    python scripts/ingest_documents.py samples/ --force-reindex

Walks the given path (file or directory), stores every supported document,
and indexes it using the configured settings (.env or environment).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.dependencies import build_container  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError  # noqa: E402
from app.ingestion.pipeline import IngestionError  # noqa: E402

logger = get_logger(__name__)


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(candidate)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG platform")
    parser.add_argument("path", type=Path, help="File or directory to ingest")
    parser.add_argument(
        "--no-index", action="store_true", help="Store documents without indexing them"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Re-index documents even if they were already indexed",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    container = build_container(settings)

    if not args.path.exists():
        print(f"error: path does not exist: {args.path}", file=sys.stderr)
        return 1

    files = collect_files(args.path)
    if not files:
        print(f"error: no supported files found under {args.path}", file=sys.stderr)
        return 1

    indexed = duplicates = failed = skipped = 0
    for file_path in files:
        try:
            data = file_path.read_bytes()
            document_id, duplicate = container.pipeline.ingest(file_path.name, data)
        except UnsupportedFileTypeError as exc:
            print(f"skip  {file_path.name}: {exc}")
            skipped += 1
            continue

        if duplicate:
            duplicates += 1
            print(f"duplicate {file_path.name} (already ingested)")

        document = container.repository.get(document_id)
        needs_index = (
            args.force_reindex or document.status == "uploaded"
        ) and not args.no_index
        if needs_index:
            try:
                chunks = container.pipeline.index_document(document_id)
                indexed += 1
                print(f"indexed  {file_path.name} ({chunks} chunks) id={document_id}")
            except IngestionError as exc:
                failed += 1
                print(f"FAILED   {file_path.name}: {exc}", file=sys.stderr)
        elif args.no_index:
            print(f"stored   {file_path.name} id={document_id}")
        else:
            print(f"cached   {file_path.name} (already indexed, id={document_id})")

    print(
        f"\nDone: {indexed} indexed, {duplicates} duplicates, {skipped} skipped, {failed} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
