"""Integration tests for the document update (PUT) workflow and cache/cost."""

from tests.integration.conftest import upload_document

ORIGINAL = "The deployment pipeline runs unit tests. " * 12
UPDATED = "The deployment pipeline now also runs security scans. " * 12


def test_replace_document_updates_content_and_reindexes(client):
    upload = upload_document(client, "notes.txt", ORIGINAL)
    document_id = upload.json()["id"]
    client.post(f"/documents/{document_id}/index")

    replace = client.put(
        f"/documents/{document_id}",
        files={"file": ("notes-v2.txt", UPDATED.encode("utf-8"), "text/plain")},
    )
    assert replace.status_code == 200, replace.text
    body = replace.json()
    assert body["name"] == "notes-v2.txt"
    assert body["status"] == "indexed"
    assert body["num_chunks"] > 0

    answer = client.post(
        "/query", json={"question": "What does the deployment pipeline run?"}
    )
    assert answer.status_code == 200
    assert "security scans" in answer.json()["answer"]


def test_replace_unknown_document_returns_404(client):
    response = client.put(
        "/documents/missing",
        files={"file": ("x.txt", b"data", "text/plain")},
    )
    assert response.status_code == 404


def test_replace_rejects_unsupported_file_type(client):
    upload = upload_document(client, "notes.txt", ORIGINAL)
    document_id = upload.json()["id"]
    response = client.put(
        f"/documents/{document_id}",
        files={"file": ("x.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_replace_conflicting_content_returns_409(client):
    first = upload_document(client, "a.txt", ORIGINAL)
    second = upload_document(client, "b.txt", "Different document entirely. " * 10)
    response = client.put(
        f"/documents/{second.json()['id']}",
        files={"file": ("b.txt", ORIGINAL.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 409
    assert first.json()["id"] != second.json()["id"]


def test_upload_with_auto_index(client):
    response = client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("instant.txt", ("Auto indexed. " * 40).encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "indexed"
    assert body["num_chunks"] > 0


def test_query_cache_hits_then_invalidates(seeded_client):
    payload = {"question": "How does the deployment pipeline validate changes?"}

    first = seeded_client.post("/query", json=payload)
    assert first.status_code == 200
    assert first.json()["cached"] is False

    second = seeded_client.post("/query", json=payload)
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["answer"] == first.json()["answer"]

    # Any corpus change (revision bump) invalidates cached answers.
    document_id = seeded_client.get("/documents").json()[0]["id"]
    assert seeded_client.delete(f"/documents/{document_id}").status_code == 204
    third = seeded_client.post("/query", json=payload)
    assert third.json()["cached"] is False


def test_query_cache_respects_filters(seeded_client):
    payload = {"question": "What is prompt injection?"}
    bare = seeded_client.post("/query", json=payload).json()

    document_id = seeded_client.get("/documents").json()[0]["id"]
    filtered = seeded_client.post(
        "/query", json={**payload, "document_ids": [document_id]}
    ).json()
    assert filtered["cached"] is False

    again = seeded_client.post(
        "/query", json={**payload, "document_ids": [document_id]}
    ).json()
    assert again["cached"] is True
    assert again["retrieval"]["chunks_returned"] == filtered["retrieval"]["chunks_returned"]
    assert bare["cached"] is False  # separate cache entry


def test_cost_tracking_reports_estimated_cost(client, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.integration.conftest import make_settings

    settings = make_settings(
        tmp_path,
        llm_cost_prompt_per_1k=0.50,
        llm_cost_completion_per_1k=1.50,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        upload = upload_document(c, "cost.txt", "Cost tracking content. " * 30)
        c.post(f"/documents/{upload.json()['id']}/index")

        response = c.post("/query", json={"question": "What does cost tracking content say?"})
        assert response.status_code == 200
        usage = response.json()["usage"]
        assert usage["cost_usd"] > 0
        expected = (
            usage["prompt_tokens"] / 1000 * 0.50
            + usage["completion_tokens"] / 1000 * 1.50
        )
        assert abs(usage["cost_usd"] - round(expected, 6)) < 1e-9

        metrics = c.get("/metrics").text
        assert "rag_llm_cost_usd_total" in metrics
        assert "rag_query_cache_total" in metrics
