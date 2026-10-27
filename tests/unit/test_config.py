"""Unit tests for settings validation and environment overrides."""

import pytest

from app.core.config import Settings


def test_defaults():
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.vector_store == "local"
    assert settings.embedding_provider == "hashing"
    assert settings.llm_provider == "stub"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("HYBRID_RETRIEVAL", "false")
    settings = Settings()
    assert settings.chunk_size == 512
    assert settings.app_env == "production"
    assert settings.hybrid_retrieval is False


def test_overlap_clamped_to_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "300")
    monkeypatch.setenv("CHUNK_OVERLAP", "5000")
    settings = Settings()
    assert 0 <= settings.chunk_overlap < settings.chunk_size


def test_invalid_chunk_size_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "10")
    with pytest.raises(ValueError):
        Settings()


def test_extension_list_parsing():
    settings = Settings(allowed_extensions="PDF, .txt, md")
    assert settings.allowed_extension_list == [".pdf", ".txt", ".md"]


def test_cors_origin_list():
    settings = Settings(cors_origins="http://a,http://b")
    assert settings.cors_origin_list == ["http://a", "http://b"]


def test_upload_limit_bytes():
    settings = Settings(max_upload_size_mb=2)
    assert settings.max_upload_size_bytes == 2 * 1024 * 1024
