"""Integration tests for the grounded question answering API."""


def test_query_returns_grounded_answer_with_citations(seeded_client):
    response = seeded_client.post(
        "/query",
        json={"question": "How does the deployment pipeline validate changes?", "top_k": 3},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["answer"]
    assert body["insufficient_context"] is False
    assert body["retrieval"]["chunks_returned"] > 0
    assert body["retrieval"]["mode"] == "hybrid"

    assert body["citations"], "stub answers should cite sources"
    citation = body["citations"][0]
    assert citation["chunk_id"]
    assert citation["document_id"]
    assert citation["page"] >= 1
    assert citation["document_name"] in {"deployment.md", "security.md"}


def test_query_without_citations_when_disabled(seeded_client):
    response = seeded_client.post(
        "/query",
        json={"question": "What is prompt injection?", "include_citations": False},
    )
    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert response.json()["retrieval"]["chunks_returned"] > 0


def test_query_document_filter_restricts_results(seeded_client):
    seeded = seeded_client.get("/documents").json()
    by_name = {doc["name"]: doc["id"] for doc in seeded}
    security_id = by_name["security.md"]

    response = seeded_client.post(
        "/query",
        json={"question": "How does the deployment pipeline work?", "document_ids": [security_id]},
    )
    assert response.status_code == 200
    retrieval = response.json()["retrieval"]
    assert retrieval["chunks_returned"] > 0
    assert all(chunk["document_id"] == security_id for chunk in retrieval["chunks"])


def test_query_unknown_filter_yields_insufficient_context(seeded_client):
    response = seeded_client.post(
        "/query",
        json={
            "question": "How does the deployment pipeline work?",
            "document_ids": ["missing-doc"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_context"] is True
    assert body["retrieval"]["chunks_returned"] == 0
    assert body["citations"] == []


def test_query_semantic_mode(seeded_client):
    response = seeded_client.post(
        "/query",
        json={"question": "What protects against prompt injection?", "mode": "semantic"},
    )
    assert response.status_code == 200
    assert response.json()["retrieval"]["mode"] == "semantic"


def test_query_validation_error(seeded_client):
    response = seeded_client.post("/query", json={"question": ""})
    assert response.status_code == 422

    response = seeded_client.post("/query", json={"question": "hi", "top_k": 0})
    assert response.status_code == 422


def test_query_reports_usage_and_latency(seeded_client):
    response = seeded_client.post("/query", json={"question": "What is rollback?"})
    body = response.json()
    assert body["usage"]["total_tokens"] >= 0
    assert body["usage"]["model"]
    assert body["latency_ms"] >= 0
    assert body["prompt_version"]
