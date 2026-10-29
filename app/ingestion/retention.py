"""Retention: delete documents that have not been touched for N days.

``RetentionSweeper`` runs a daemon thread that sweeps periodically; the
same sweep is available on demand via POST /maintenance/retention. A
document's age resets whenever it is (re)indexed or replaced, so actively
updated content is never removed.
"""

import threading
from datetime import UTC, datetime, timedelta

from app.core.logging import get_logger
from app.ingestion.pipeline import DocumentNotFoundError, IngestionPipeline

logger = get_logger(__name__)


def sweep_retention(pipeline: IngestionPipeline, retention_days: int) -> list[str]:
    """Delete documents older than ``retention_days``; returns deleted ids."""

    if retention_days <= 0:
        return []
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    expired = pipeline.repository.list_expired(cutoff)
    deleted: list[str] = []
    for document in expired:
        try:
            pipeline.delete_document(document.id)
            deleted.append(document.id)
        except DocumentNotFoundError:  # pragma: no cover - raced with a delete
            continue
    if deleted:
        logger.info(
            "Retention sweep deleted documents",
            extra={"deleted": len(deleted), "retention_days": retention_days},
        )
    return deleted


class RetentionSweeper:
    """Background thread that applies the retention policy on an interval."""

    def __init__(
        self, pipeline: IngestionPipeline, retention_days: int, interval_seconds: int
    ) -> None:
        self.pipeline = pipeline
        self.retention_days = retention_days
        self.interval_seconds = max(60, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="retention", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                sweep_retention(self.pipeline, self.retention_days)
            except Exception:  # pragma: no cover - never kill the sweeper
                logger.exception("Retention sweep failed")
