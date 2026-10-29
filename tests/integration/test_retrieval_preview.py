"""Integration tests for the retrieval preview / debugger endpoint."""

import io


def test_retrieval_preview_hybrid_returns_vector_and_bm25(client):
    content = b"Kubernetes uses pods and deployments. Postgres stores relational tables."
    upload = client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("tech.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload.status_code == 201

    response = client.post(
        "/retrieval/preview",
        json={
            "question": "What stores relational tables?",
            "top_k": 3,
            "mode": "hybrid",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert "latency_ms" in data
    assert len(data["fused_chunks"]) > 0
    assert len(data["vector_hits"]) > 0
    assert len(data["bm25_hits"]) > 0

    first_hit = data["vector_hits"][0]
    assert "chunk_id" in first_hit
    assert "score" in first_hit
    assert first_hit["rank"] == 1


def test_retrieval_preview_semantic_only(client):
    content = b"Kubernetes uses pods and deployments for container workloads."
    upload = client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("k8s.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload.status_code == 201

    response = client.post(
        "/retrieval/preview",
        json={
            "question": "Kubernetes pods",
            "top_k": 2,
            "mode": "semantic",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "semantic"
    assert len(data["vector_hits"]) > 0
    assert len(data["bm25_hits"]) == 0
