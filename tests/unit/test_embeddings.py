"""Unit tests for embedding providers."""

import math

import pytest

from app.core.config import Settings
from app.retrieval.embeddings import (
    EmbeddingError,
    HashingEmbeddings,
    build_embedding_provider,
)


def test_hashing_embeddings_deterministic():
    provider = HashingEmbeddings(dimension=128)
    first = provider.embed_query("the quick brown fox")
    second = provider.embed_query("the quick brown fox")
    assert first == second


def test_hashing_embeddings_normalized():
    provider = HashingEmbeddings(dimension=128)
    vector = provider.embed_query("normalize me please")
    norm = math.sqrt(sum(x * x for x in vector))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_hashing_embeddings_dimension():
    provider = HashingEmbeddings(dimension=256)
    vectors = provider.embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == 256 for v in vectors)


def test_similar_texts_score_higher_than_unrelated():
    provider = HashingEmbeddings(dimension=256)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    doc = provider.embed_query("docker containers package applications for deployment")
    similar = provider.embed_query("docker containers for deployment")
    unrelated = provider.embed_query("quarterly financial revenue report")
    assert dot(doc, similar) > dot(doc, unrelated)


def test_hashing_rejects_tiny_dimension():
    with pytest.raises(ValueError):
        HashingEmbeddings(dimension=10)


def test_build_provider_default_is_hashing():
    provider = build_embedding_provider(Settings())
    assert isinstance(provider, HashingEmbeddings)


def test_build_provider_unknown_raises():
    with pytest.raises(ValueError):
        build_embedding_provider(Settings(embedding_provider="nope"))


def test_openai_provider_requires_key(monkeypatch):
    from app.retrieval.embeddings import OpenAIEmbeddings

    provider = OpenAIEmbeddings(model="text-embedding-3-small", api_key=None)
    with pytest.raises(EmbeddingError):
        provider.embed_query("hello")
