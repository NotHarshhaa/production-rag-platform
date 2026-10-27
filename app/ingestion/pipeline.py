"""Ingestion pipeline: persist uploads, extract text, chunk, embed, and index."""

import hashlib
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.logging import get_logger
from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import detect_content_type, load_document
from app.observability.metrics import (
    DOCUMENTS_INDEXED,
    INGESTION_LATENCY,
)
from app.retrieval.vector_store import VectorRecord, VectorStore
from app.storage.repository import DocumentRepository

logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when a document cannot be indexed."""


class DocumentNotFoundError(KeyError):
    """Raised when an operation references an unknown document id."""


@dataclass
class IngestResult:
    document_id: str
    name: str
    status: str
    duplicate: bool


class IngestionPipeline:
    def __init__(
        self,
        repository: DocumentRepository,
        embeddings,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.settings = settings

    def ingest(self, name: str, data: bytes) -> tuple[str, bool]:
        """Persist an upload. Returns (document_id, is_duplicate).

        Duplicate detection is content-based: a byte-identical file resolves
        to the existing document instead of creating a new row.
        """

        sha256 = hashlib.sha256(data).hexdigest()
        existing = self.repository.get_by_sha256(sha256)
        if existing is not None:
            logger.info(
                "Duplicate upload detected",
                extra={"file": name, "existing_document_id": existing.id},
            )
            return existing.id, True

        content_type = detect_content_type(name)
        document = self.repository.create(
            name=name,
            content_type=content_type,
            size_bytes=len(data),
            sha256=sha256,
            content=data,
        )
        logger.info("Document stored", extra={"document_id": document.id, "file": name})
        return document.id, False

    def index_document(self, document_id: str) -> int:
        """Index or re-index a stored document. Returns the chunk count.

        Re-indexing is idempotent: existing vectors for the document are
        replaced atomically from the client's point of view.
        """

        started = time.perf_counter()
        document = self.repository.get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)

        try:
            loaded = load_document(document.name, document.content)
            chunks = chunk_document(
                loaded, self.settings.chunk_size, self.settings.chunk_overlap
            )
            if not chunks:
                raise IngestionError(
                    "No extractable text found; the document cannot be indexed"
                )

            vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
            self.vector_store.delete_document(document_id)
            records = [
                VectorRecord(
                    id=f"{document_id}:{chunk.index}",
                    vector=vector,
                    metadata={
                        "chunk_id": f"{document_id}:{chunk.index}",
                        "chunk_index": chunk.index,
                        "document_id": document_id,
                        "document_name": document.name,
                        "content_type": document.content_type,
                        "page": chunk.page,
                        "text": chunk.text,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            self.vector_store.add(records)

            self.repository.update_status(
                document_id, status="indexed", num_chunks=len(chunks)
            )
            duration = time.perf_counter() - started
            INGESTION_LATENCY.labels(operation="index").observe(duration)
            DOCUMENTS_INDEXED.labels(result="indexed").inc()
            logger.info(
                "Document indexed",
                extra={
                    "document_id": document_id,
                    "chunks": len(chunks),
                    "duration_ms": round(duration * 1000, 2),
                },
            )
            return len(chunks)
        except IngestionError:
            self.repository.update_status(
                document_id, status="failed", num_chunks=0, error="no extractable text"
            )
            DOCUMENTS_INDEXED.labels(result="failed").inc()
            raise
        except Exception as exc:
            self.repository.update_status(
                document_id, status="failed", num_chunks=0, error=str(exc)
            )
            DOCUMENTS_INDEXED.labels(result="failed").inc()
            logger.exception("Indexing failed", extra={"document_id": document_id})
            raise IngestionError(f"Indexing failed: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        """Remove a document's vectors and metadata row."""

        document = self.repository.get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        self.vector_store.delete_document(document_id)
        self.repository.delete(document_id)
        logger.info("Document deleted", extra={"document_id": document_id})
