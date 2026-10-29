"""Integration tests for multi-turn conversation memory."""

import io


def test_query_with_conversation_history(client):
    content = b"The production platform uses PostgreSQL for metadata and Qdrant for vectors."
    client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("setup.txt", io.BytesIO(content), "text/plain")},
    )

    history = [
        {"role": "user", "content": "What is this platform?"},
        {"role": "assistant", "content": "It is a production RAG platform."},
    ]
    response = client.post(
        "/query",
        json={
            "question": "What databases does it use?",
            "conversation_history": history,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert len(body["retrieval"]["chunks"]) > 0


def test_query_stream_with_conversation_history(client):
    content = b"Deployment uses Docker Compose and Kubernetes manifests."
    client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("deploy.txt", io.BytesIO(content), "text/plain")},
    )

    history = [
        {"role": "user", "content": "Tell me about deployment."},
        {"role": "assistant", "content": "Deployment is containerized."},
    ]
    response = client.post(
        "/query/stream",
        json={
            "question": "How does it scale on Kubernetes?",
            "conversation_history": history,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
