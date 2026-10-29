"""Hybrid retrieval: dense vector search + BM25 keyword search, fused with RRF.

The BM25 index is rebuilt lazily whenever the vector store's revision
counter changes, so newly ingested documents are searchable immediately.
Note the revision counter is process-local: run the API with a single
worker (the default) when HYBRID_RETRIEVAL is enabled.

``mode="agentic"`` decomposes the question into sub-queries (see
app/retrieval/query_decomposer.py), retrieves for each, and fuses the
result lists with RRF — better coverage for compound questions.
"""

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.ingestion.chunking import tokenize
from app.observability.metrics import RETRIEVAL_LATENCY, RETRIEVED_CHUNKS
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.query_decomposer import decompose_question
from app.retrieval.vector_store import VectorStore, matches_filter

logger = get_logger(__name__)

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    score: float
    sources: set[str] = field(default_factory=set)


class BM25Index:
    """Okapi BM25 over the indexed chunks. Pure Python, no extra dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._keys: list[str] = []
        self._doc_term_counts: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._doc_metadata: list[dict[str, Any]] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self._keys)

    def build(self, documents: list[tuple[str, str, dict[str, Any]]]) -> None:
        """Build the index from (key, text, metadata) tuples."""

        self._keys, self._doc_term_counts, self._doc_lengths, self._doc_metadata = [], [], [], []
        self._doc_freqs = {}
        total_length = 0
        for key, text, metadata in documents:
            tokens = tokenize(text)
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            for term in term_counts:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1
            self._keys.append(key)
            self._doc_term_counts.append(term_counts)
            self._doc_lengths.append(len(tokens))
            self._doc_metadata.append(metadata)
            total_length += len(tokens)
        self._avg_length = (total_length / len(self._keys)) if self._keys else 0.0

    def _idf(self, term: str) -> float:
        doc_freq = self._doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0.0
        n = len(self._keys)
        return ((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).__float__()

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._keys or top_k <= 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[float, int]] = []
        for position, term_counts in enumerate(self._doc_term_counts):
            if not matches_filter(self._doc_metadata[position], metadata_filter):
                continue
            score = 0.0
            avg_length = self._avg_length or 1.0
            length_norm = self.k1 * (
                1 - self.b + self.b * self._doc_lengths[position] / avg_length
            )
            for term in query_terms:
                freq = term_counts.get(term)
                if not freq:
                    continue
                score += self._idf(term) * (freq * (self.k1 + 1)) / (freq + length_norm)
            if score > 0:
                scores.append((score, position))

        top = heapq.nlargest(top_k, scores, key=lambda item: item[0])
        return [(self._keys[position], score) for score, position in top]


class HybridRetriever:
    """Dense, hybrid, agentic, and (optionally) reranked retrieval."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings,
        llm=None,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.settings = settings
        self.llm = llm
        self._bm25 = BM25Index()
        self._bm25_revision: int = -1
        self._metadata_by_key: dict[str, dict[str, Any]] = {}
        self._index_lock = threading.Lock()

    def _ensure_bm25_index(self) -> BM25Index:
        revision = self.vector_store.get_revision()
        if revision != self._bm25_revision:
            with self._index_lock:
                revision = self.vector_store.get_revision()
                if revision != self._bm25_revision:
                    metadata_by_key: dict[str, dict[str, Any]] = {}
                    documents = []
                    for record in self.vector_store.iter_chunks():
                        metadata_by_key[record.id] = record.metadata
                        documents.append(
                            (record.id, record.metadata.get("text", ""), record.metadata)
                        )
                    self._bm25.build(documents)
                    self._metadata_by_key = metadata_by_key
                    self._bm25_revision = revision
                    logger.debug("BM25 index rebuilt", extra={"documents": len(documents)})
        return self._bm25

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval_top_k
        retrieval_mode = mode or ("hybrid" if self.settings.hybrid_retrieval else "semantic")

        started = time.perf_counter()
        if retrieval_mode == "agentic":
            chunks = self._retrieve_agentic(question, top_k, metadata_filter)
        elif retrieval_mode == "hybrid":
            chunks = self._retrieve_hybrid(question, top_k, metadata_filter)
        else:
            chunks = self._retrieve_semantic(question, top_k, metadata_filter)
        duration = time.perf_counter() - started

        RETRIEVAL_LATENCY.observe(duration)
        RETRIEVED_CHUNKS.labels(mode=retrieval_mode).observe(len(chunks))
        return chunks

    def subqueries_for(self, question: str) -> list[str]:
        """The sub-queries agentic mode would run for ``question``."""

        return decompose_question(
            question, llm=self.llm, max_subqueries=self.settings.agentic_max_subqueries
        )

    def _retrieve_agentic(
        self, question: str, top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        subqueries = self.subqueries_for(question)
        candidate_k = max(top_k, self.settings.retrieval_candidate_k)
        per_query_limit = min(candidate_k, max(top_k, top_k * 2))

        fused_scores: dict[str, float] = {}
        chunks_by_key: dict[str, RetrievedChunk] = {}
        for subquery in subqueries:
            if self.settings.hybrid_retrieval:
                hits = self._retrieve_hybrid(subquery, per_query_limit, metadata_filter)
            else:
                hits = self._retrieve_semantic(subquery, per_query_limit, metadata_filter)
            for rank, chunk in enumerate(hits, start=1):
                key = chunk.chunk_id
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
                existing = chunks_by_key.get(key)
                if existing is None:
                    chunks_by_key[key] = chunk
                else:
                    existing.sources |= chunk.sources

        ordered_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
        results = []
        for key in ordered_keys[:top_k]:
            chunk = chunks_by_key[key]
            chunk.score = round(fused_scores[key], 6)
            results.append(chunk)
        if results:
            logger.debug(
                "Agentic retrieval fused sub-queries",
                extra={"subqueries": len(subqueries), "fused_chunks": len(results)},
            )
        return results

    def _retrieve_semantic(
        self, question: str, top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed_query(question)
        hits = self.vector_store.search(query_vector, top_k, metadata_filter)
        chunks = [self._to_chunk(hit.id, hit.score, hit.metadata, {"vector"}) for hit in hits]
        if self.settings.min_relevance_score > 0:
            chunks = [c for c in chunks if c.score >= self.settings.min_relevance_score]
        return chunks

    def _retrieve_hybrid(
        self, question: str, top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        candidate_k = max(top_k, self.settings.retrieval_candidate_k)
        query_vector = self.embeddings.embed_query(question)
        vector_hits = self.vector_store.search(query_vector, candidate_k, metadata_filter)
        bm25_hits = self._ensure_bm25_index().search(question, candidate_k, metadata_filter)

        fused: dict[str, float] = {}
        metadata_by_key: dict[str, dict[str, Any]] = {}
        sources_by_key: dict[str, set[str]] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (RRF_K + rank)
            metadata_by_key[hit.id] = hit.metadata
            sources_by_key.setdefault(hit.id, set()).add("vector")
        # Refresh the BM25 index (and its metadata cache) before fusing.
        self._ensure_bm25_index()
        for rank, (key, _score) in enumerate(bm25_hits, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in metadata_by_key:
                # BM25 hit outside the vector top-N: metadata is cached from
                # the last index rebuild.
                metadata = self._metadata_by_key.get(key)
                if metadata is not None:
                    metadata_by_key[key] = metadata
            if key in metadata_by_key:
                sources_by_key.setdefault(key, set()).add("bm25")

        ordered_keys = sorted(fused, key=lambda key: fused[key], reverse=True)[:top_k]
        return [
            self._to_chunk(key, fused[key], metadata_by_key[key], sources_by_key[key])
            for key in ordered_keys
            if key in metadata_by_key
        ]

    @staticmethod
    def _to_chunk(
        key: str,
        score: float,
        metadata: dict[str, Any],
        sources: set[str],
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=metadata.get("chunk_id", key),
            document_id=metadata.get("document_id", ""),
            document_name=metadata.get("document_name", ""),
            page=int(metadata.get("page", 0) or 0),
            text=metadata.get("text", ""),
            score=round(float(score), 6),
            sources=sources,
        )
    def debug_retrieve(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        retrieval_mode = mode or ("hybrid" if self.settings.hybrid_retrieval else "semantic")
        candidate_k = max(top_k, self.settings.retrieval_candidate_k)

        query_vector = self.embeddings.embed_query(question)
        vector_hits_raw = self.vector_store.search(query_vector, candidate_k, metadata_filter)
        vector_hits = [
            {
                "chunk_id": hit.metadata.get("chunk_id", hit.id),
                "document_id": hit.metadata.get("document_id", ""),
                "document_name": hit.metadata.get("document_name", ""),
                "page": int(hit.metadata.get("page", 1) or 1),
                "score": round(float(hit.score), 6),
                "rank": rank,
                "text": hit.metadata.get("text", ""),
            }
            for rank, hit in enumerate(vector_hits_raw, start=1)
        ]

        bm25_hits: list[dict[str, Any]] = []
        subqueries: list[str] = []
        if retrieval_mode == "hybrid" or retrieval_mode == "agentic":
            if retrieval_mode == "agentic":
                subqueries = self.subqueries_for(question)
            bm25_raw = self._ensure_bm25_index().search(question, candidate_k, metadata_filter)
            meta_lookup = {hit.id: hit.metadata for hit in vector_hits_raw}
            for rank, (key, score) in enumerate(bm25_raw, start=1):
                meta = meta_lookup.get(key) or self._metadata_by_key.get(key)
                meta = meta or {}
                bm25_hits.append(
                    {
                        "chunk_id": meta.get("chunk_id", key),
                        "document_id": meta.get("document_id", ""),
                        "document_name": meta.get("document_name", ""),
                        "page": int(meta.get("page", 1) or 1),
                        "score": round(float(score), 6),
                        "rank": rank,
                        "text": meta.get("text", ""),
                    }
                )

        fused_chunks = self.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter, mode=mode
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        return {
            "question": question,
            "mode": retrieval_mode,
            "latency_ms": latency_ms,
            "fused_chunks": fused_chunks,
            "vector_hits": vector_hits[:top_k],
            "bm25_hits": bm25_hits[:top_k],
            "subqueries": subqueries,
        }

