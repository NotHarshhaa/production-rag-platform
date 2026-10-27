"""Integration tests for background ingestion workers."""

import time


def _await_status(client, document_id, expected, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        document = client.get(f"/documents/{document_id}").json()
        if document["status"] == expected:
            return document
        time.sleep(0.1)
    raise AssertionError(f"document {document_id} never reached status {expected!r}")


def test_index_endpoint_queues_job_and_returns_202(background_client):
    content = ("Background indexing test. " * 40).encode("utf-8")
    upload = background_client.post(
        "/documents/upload",
        files={"file": ("async.txt", content, "text/plain")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["id"]
    assert upload.json()["status"] == "uploaded"

    response = background_client.post(f"/documents/{document_id}/index")
    assert response.status_code == 202

    document = _await_status(background_client, document_id, "indexed")
    assert document["num_chunks"] > 0


def test_upload_auto_index_queues_job(background_client):
    content = ("Queued upload test. " * 40).encode("utf-8")
    upload = background_client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("queued.txt", content, "text/plain")},
    )
    assert upload.status_code == 202
    document_id = upload.json()["id"]
    assert upload.json()["status"] == "uploaded"

    document = _await_status(background_client, document_id, "indexed")
    assert document["num_chunks"] > 0

    answer = background_client.post(
        "/query", json={"question": "What does the queued upload test say?"}
    )
    assert answer.status_code == 200
    assert answer.json()["retrieval"]["chunks_returned"] > 0


def test_background_failure_marks_document_failed(background_client):
    upload = background_client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    document_id = upload.json()["id"]
    background_client.post(f"/documents/{document_id}/index")
    document = _await_status(background_client, document_id, "failed")
    assert document["error"]


def test_unknown_document_background_index_is_404(background_client):
    response = background_client.post("/documents/nonexistent/index")
    assert response.status_code == 404
