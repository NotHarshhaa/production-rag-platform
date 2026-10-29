"""Integration tests for the answer groundedness score feature."""

import io


def test_query_returns_groundedness_score(client):
    content = b"The platform supports hybrid BM25 and vector retrieval with RRF fusion."
    upload = client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("grounded.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload.status_code == 201

    response = client.post(
        "/query",
        json={"question": "What retrieval does the platform support?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "groundedness_score" in data
    assert 0.0 <= data["groundedness_score"] <= 1.0


def test_query_insufficient_context_has_full_groundedness(client):
    response = client.post(
        "/query",
        json={"question": "What is quantum gravity string theory?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["insufficient_context"] is True
    assert data["groundedness_score"] == 1.0
