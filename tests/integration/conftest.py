"""Shared fixtures: a fully wired test app with stub providers."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

DOC_ONE = """# Deployment Pipeline

The deployment pipeline validates changes before they reach production.
Every change runs through the CI stage, which executes unit tests and
integration tests. After the build stage produces a Docker image, the deploy
stage promotes it to staging and then to production with a blue-green
release. Rollback re-points the load balancer at the previous image tag.
""" * 3

DOC_TWO = """# Security Controls

Prompt injection is a RAG-specific risk: documents are untrusted input and
may contain embedded instructions. Upload validation enforces an allow list
of file types and size limits. Credentials are stored in environment
variables, never in source control. Retrieval filters by document ownership
so tenant data stays isolated.
""" * 3


def make_settings(tmp_path, **overrides) -> Settings:
    values = dict(
        database_url=f"sqlite:///{tmp_path / 'rag-test.db'}",
        vector_store="inmemory",
        embedding_provider="hashing",
        embedding_dimension=128,
        llm_provider="stub",
        chunk_size=400,
        chunk_overlap=50,
        retrieval_top_k=3,
        retrieval_candidate_k=10,
        metrics_enabled=True,
        log_level="WARNING",
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture()
def client(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def seeded_client(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as test_client:
        for filename, content in (("deployment.md", DOC_ONE), ("security.md", DOC_TWO)):
            response = test_client.post(
                "/documents/upload",
                files={"file": (filename, content.encode("utf-8"), "text/markdown")},
            )
            assert response.status_code == 201, response.text
            document_id = response.json()["id"]
            index_response = test_client.post(f"/documents/{document_id}/index")
            assert index_response.status_code == 200, index_response.text
        yield test_client


def upload_document(client: TestClient, filename: str, content: str):
    return client.post(
        "/documents/upload",
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )


@pytest.fixture()
def auth_client(tmp_path):
    """App with API-key auth enabled: two tenants sharing one corpus config."""

    settings = make_settings(
        tmp_path,
        auth_enabled=True,
        api_keys="sk-acme:acme,sk-globex:globex",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def background_client(tmp_path):
    settings = make_settings(tmp_path, ingestion_mode="background")
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def upload_as(client: TestClient, filename: str, content: str, headers: dict | None = None):
    return client.post(
        "/documents/upload",
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
        headers=headers or {},
    )
