"""Embedding providers behind a common interface.

Three implementations ship with the platform:

- ``HashingEmbeddings`` — dependency-free, deterministic feature hashing.
  Useful for tests, evaluation smoke runs, and offline development.
- ``SentenceTransformerEmbeddings`` — local models via sentence-transformers
  (install the ``embeddings-local`` extra).
- ``OpenAIEmbeddings`` — any OpenAI-compatible embeddings API.

Add new providers by subclassing ``EmbeddingProvider`` and registering them
in ``build_embedding_provider``.
"""

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from app.core.config import (
    EMBEDDING_HASHING,
    EMBEDDING_OPENAI,
    EMBEDDING_SENTENCE_TRANSFORMERS,
    Settings,
)
from app.core.logging import get_logger
from app.ingestion.chunking import tokenize

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce vectors."""


class EmbeddingProvider(ABC):
    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""

    def _validate(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        if vectors and len(vectors[0]) != expected:
            raise EmbeddingError(
                f"{self.name}: embedding dimension {len(vectors[0])} != expected {expected}"
            )
        return vectors


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class HashingEmbeddings(EmbeddingProvider):
    """Deterministic feature-hashing embeddings (the "sign hashing" trick).

    Unigram and bigram features are hashed into a fixed-size vector and
    L2-normalized, which produces stable vectors across processes and
    platforms without any model download.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("hashing dimension must be at least 64")
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "little") % self.dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        return index, sign

    def _features(self, text: str) -> dict[int, float]:
        tokens = tokenize(text)
        counts: dict[int, float] = {}
        for position, token in enumerate(tokens):
            index, sign = self._hash_feature(token)
            counts[index] = counts.get(index, 0.0) + sign
            if position + 1 < len(tokens):
                bigram = f"{token}_{tokens[position + 1]}"
                index, sign = self._hash_feature(bigram)
                counts[index] = counts.get(index, 0.0) + sign * 0.7
        return counts

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for index, weight in self._features(text).items():
            vector[index] += weight
        return _normalize(vector).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._embed(text) for text in texts]
        return self._validate(vectors, self.dimension)

    def embed_query(self, text: str) -> list[float]:
        return self._validate([self._embed(text)], self.dimension)[0]


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local embeddings via sentence-transformers (lazy model load)."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.batch_size = batch_size
        self.name = self._model_name
        self._model = None
        self.dimension = 0

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    'Install it with: pip install -e ".[embeddings-local]"'
                ) from exc
            logger.info("Loading sentence-transformers model", extra={"model": self._model_name})
            self._model = SentenceTransformer(self._model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return self._validate(np.asarray(embeddings).tolist(), self.dimension)

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        embedding = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return self._validate(np.asarray(embedding).tolist(), self.dimension)[0]


class OpenAIEmbeddings(EmbeddingProvider):
    """Embeddings from any OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self._model = model or "text-embedding-3-small"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = self._model
        self.dimension = 0

    def _request(self, input_texts: list[str]) -> list[list[float]]:
        import httpx

        if not self.api_key:
            raise EmbeddingError("LLM_API_KEY is required for the openai embedding provider")
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self._model, "input": input_texts},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise EmbeddingError(
                f"Embedding request failed with status {response.status_code}: "
                f"{response.text[:200]}"
            )
        data = response.json()["data"]
        vectors = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        self.dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._request(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower()
    if provider == EMBEDDING_HASHING:
        return HashingEmbeddings(dimension=settings.embedding_dimension)
    if provider == EMBEDDING_SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbeddings(settings.embedding_model)
    if provider == EMBEDDING_OPENAI:
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}. "
        f"Supported: {EMBEDDING_HASHING}, {EMBEDDING_SENTENCE_TRANSFORMERS}, {EMBEDDING_OPENAI}"
    )
