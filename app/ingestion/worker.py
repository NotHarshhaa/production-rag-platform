"""Background ingestion workers.

When INGESTION_MODE=background, indexing runs on a small thread pool and the
API returns 202 Accepted immediately; clients poll GET /documents/{id} for
status transitions (uploaded -> indexed/failed).
"""

from concurrent.futures import ThreadPoolExecutor

from app.core.logging import get_logger
from app.ingestion.pipeline import DocumentNotFoundError, IngestionError, IngestionPipeline

logger = get_logger(__name__)


class BackgroundIndexer:
    """Bounded worker pool for asynchronous document indexing."""

    def __init__(self, pipeline: IngestionPipeline, workers: int = 2) -> None:
        self.pipeline = pipeline
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="ingest")
        self.queued = 0

    def submit(self, document_id: str) -> None:
        self.queued += 1
        self._pool.submit(self._run, document_id)
        logger.info("Indexing queued", extra={"document_id": document_id})

    def _run(self, document_id: str) -> None:
        try:
            self.pipeline.index_document(document_id)
        except DocumentNotFoundError:
            logger.error("Queued document disappeared", extra={"document_id": document_id})
        except IngestionError as exc:
            logger.error(
                "Background indexing failed",
                extra={"document_id": document_id, "error": str(exc)},
            )
        finally:
            self.queued = max(0, self.queued - 1)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
