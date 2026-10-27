"""Optional reranking of retrieved chunks.

The cross-encoder uses sentence-transformers (install the
``embeddings-local`` extra). When reranking is disabled or the model cannot
be loaded, a pass-through reranker keeps the pipeline working.
"""

from abc import ABC, abstractmethod

from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.hybrid_search import RetrievedChunk

logger = get_logger(__name__)


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, question: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top_k chunks, reordered by relevance."""

    @property
    def enabled(self) -> bool:
        return False


class NoOpReranker(Reranker):
    def rerank(
        self, question: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model = None
        self._warned = False

    @property
    def enabled(self) -> bool:
        return True

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                if not self._warned:
                    logger.warning(
                        "Reranking requested but sentence-transformers is not installed; "
                        "falling back to pass-through. Install with: "
                        'pip install -e ".[embeddings-local]"'
                    )
                    self._warned = True
                raise
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, question: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            model = self._get_model()
        except Exception:
            return chunks[:top_k]

        pairs = [[question, chunk.text] for chunk in chunks]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(chunks, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
        )
        reranked = []
        for chunk, score in ranked[:top_k]:
            chunk.score = round(float(score), 6)
            reranked.append(chunk)
        return reranked


def build_reranker(settings: Settings) -> Reranker:
    if not settings.rerank_enabled:
        return NoOpReranker()
    logger.info("Reranking enabled", extra={"model": settings.rerank_model})
    return CrossEncoderReranker(settings.rerank_model)
