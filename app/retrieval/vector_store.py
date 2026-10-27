"""Vector store abstraction with local (file-backed), in-memory, and Qdrant
implementations.

All implementations store the chunk text and metadata alongside the vector
so retrieval never needs a secondary lookup. Add another backend (e.g.
pgvector) by subclassing ``VectorStore`` and registering it in
``build_vector_store``.
"""

import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import (
    VECTOR_STORE_INMEMORY,
    VECTOR_STORE_LOCAL,
    VECTOR_STORE_QDRANT,
    Settings,
)

logger = logging.getLogger(__name__)

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "production-rag-platform/chunk")


class VectorStoreError(RuntimeError):
    """Raised when a vector store backend cannot be used."""


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    """Equality filter: a list value means "any of"."""

    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class VectorStore(ABC):
    revision: int = 0
    """Monotonic counter bumped on every mutation; used to invalidate the BM25 index."""

    @abstractmethod
    def add(self, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def iter_chunks(self, document_id: str | None = None) -> Iterator[VectorRecord]: ...

    @abstractmethod
    def count_chunks(self) -> int: ...

    @abstractmethod
    def get_revision(self) -> int: ...

    def clear(self) -> None:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """NumPy-backed cosine similarity store for development and tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self.revision = 0

    def get_revision(self) -> int:
        return self.revision

    def add(self, records: list[VectorRecord]) -> None:
        with self._lock:
            for record in records:
                if record.id not in self._vectors:
                    self._ids.append(record.id)
                self._vectors[record.id] = np.asarray(record.vector, dtype=np.float32)
                self._metadata[record.id] = dict(record.metadata)
            self._matrix = None
            self.revision += 1

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            doomed = [
                record_id
                for record_id in self._ids
                if self._metadata.get(record_id, {}).get("document_id") == document_id
            ]
            for record_id in doomed:
                self._ids.remove(record_id)
                self._vectors.pop(record_id, None)
                self._metadata.pop(record_id, None)
            self._matrix = None
            self.revision += 1

    def _rebuild_matrix(self) -> None:
        if self._matrix is None:
            if self._ids:
                self._matrix = np.vstack([self._vectors[record_id] for record_id in self._ids])
            else:
                self._matrix = np.zeros((0, 1), dtype=np.float32)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        with self._lock:
            self._rebuild_matrix()
            ids = list(self._ids)
            matrix = self._matrix
        if matrix is None or not ids:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        corpus_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        corpus_norm[corpus_norm == 0] = 1.0
        scores = (matrix / corpus_norm) @ query

        results: list[SearchResult] = []
        for record_id, score in zip(ids, scores.tolist(), strict=True):
            metadata = self._metadata.get(record_id, {})
            if not matches_filter(metadata, metadata_filter):
                continue
            results.append(SearchResult(id=record_id, score=float(score), metadata=metadata))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def iter_chunks(self, document_id: str | None = None) -> Iterator[VectorRecord]:
        for record_id in list(self._ids):
            metadata = self._metadata.get(record_id)
            if metadata is None:
                continue
            if document_id and metadata.get("document_id") != document_id:
                continue
            yield VectorRecord(id=record_id, vector=[], metadata=metadata)

    def count_chunks(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()
            self._metadata.clear()
            self._ids.clear()
            self._matrix = None
            self.revision += 1


class LocalVectorStore(InMemoryVectorStore):
    """In-memory cosine store persisted to a JSON file.

    The development default: same behaviour as ``InMemoryVectorStore`` but
    vectors survive process restarts, so CLI workflows (ingest in one
    command, query or evaluate in another) share state. For production use
    Qdrant — writes are synchronous JSON here, not optimized for scale.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "Vector store file is corrupted; starting empty",
                extra={"path": str(self.path)},
            )
            return
        for record in records:
            self._ids.append(record["id"])
            self._vectors[record["id"]] = np.asarray(record["vector"], dtype=np.float32)
            self._metadata[record["id"]] = record["metadata"]
        self._matrix = None
        self.revision += 1

    def _save(self) -> None:
        payload = [
            {
                "id": record_id,
                "vector": self._vectors[record_id].tolist(),
                "metadata": self._metadata[record_id],
            }
            for record_id in self._ids
        ]
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def add(self, records: list[VectorRecord]) -> None:
        super().add(records)
        with self._lock:
            self._save()

    def delete_document(self, document_id: str) -> None:
        super().delete_document(document_id)
        with self._lock:
            self._save()

    def clear(self) -> None:
        super().clear()
        with self._lock:
            self._save()


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store for production deployments."""

    def __init__(
        self, url: str, collection: str, dimension: int, api_key: str | None = None
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - qdrant-client is a core dependency
            raise VectorStoreError(
                "qdrant-client is not installed. Install it with: pip install qdrant-client"
            ) from exc

        self.collection = collection
        self.dimension = dimension
        self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._lock = threading.Lock()
        self.revision = 0
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models as qmodels

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension, distance=qmodels.Distance.COSINE
                ),
            )
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name="document_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # pragma: no cover - index is an optimization
                pass

    def get_revision(self) -> int:
        return self.revision

    def add(self, records: list[VectorRecord]) -> None:
        from qdrant_client import models as qmodels

        if not records:
            return
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(NAMESPACE, record.id)),
                vector=record.vector,
                payload={**record.metadata, "chunk_key": record.id},
            )
            for record in records
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)
        with self._lock:
            self.revision += 1

    def delete_document(self, document_id: str) -> None:
        from qdrant_client import models as qmodels

        self._client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id", match=qmodels.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )
        with self._lock:
            self.revision += 1

    def _build_filter(self, metadata_filter: dict[str, Any] | None):
        from qdrant_client import models as qmodels

        if not metadata_filter:
            return None
        must = []
        for key, expected in metadata_filter.items():
            values = list(expected) if isinstance(expected, (list, tuple, set)) else [expected]
            must.append(
                qmodels.FieldCondition(
                    key=key, match=qmodels.MatchAny(any=[str(v) for v in values])
                )
            )
        return qmodels.Filter(must=must)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        response = self._client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            limit=top_k,
            query_filter=self._build_filter(metadata_filter),
            with_payload=True,
        )
        return [
            SearchResult(
                id=point.payload.get("chunk_key", str(point.id)),
                score=float(point.score),
                metadata=dict(point.payload),
            )
            for point in response.points
        ]

    def iter_chunks(self, document_id: str | None = None) -> Iterator[VectorRecord]:
        from qdrant_client import models as qmodels

        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=(
                    qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                    if document_id
                    else None
                ),
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for point in batch:
                yield VectorRecord(
                    id=point.payload.get("chunk_key", str(point.id)),
                    vector=[],
                    metadata=dict(point.payload),
                )
            if offset is None:
                return

    def count_chunks(self) -> int:
        return self._client.count(collection_name=self.collection, exact=True).count

    def clear(self) -> None:
        from qdrant_client import models as qmodels

        self._client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(filter=qmodels.Filter()),
            wait=True,
        )
        with self._lock:
            self.revision += 1


def build_vector_store(settings: Settings, dimension: int) -> VectorStore:
    kind = settings.vector_store.lower()
    if kind == VECTOR_STORE_LOCAL:
        return LocalVectorStore(settings.vector_store_path)
    if kind == VECTOR_STORE_INMEMORY:
        return InMemoryVectorStore()
    if kind == VECTOR_STORE_QDRANT:
        if not settings.qdrant_url:
            raise VectorStoreError("QDRANT_URL is required when VECTOR_STORE=qdrant")
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimension=dimension,
            api_key=settings.qdrant_api_key or None,
        )
    raise ValueError(
        f"Unknown VECTOR_STORE {settings.vector_store!r}. "
        f"Supported: {VECTOR_STORE_LOCAL}, {VECTOR_STORE_INMEMORY}, {VECTOR_STORE_QDRANT}"
    )
