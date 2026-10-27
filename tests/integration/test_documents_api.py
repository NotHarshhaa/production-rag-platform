"""Integration tests for the documents API lifecycle."""

from tests.integration.conftest import upload_document

SAMPLE = "The deployment pipeline runs tests. " * 30


def test_upload_stores_document(client):
    response = upload_document(client, "notes.txt", SAMPLE)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "notes.txt"
    assert body["status"] == "uploaded"
    assert body["num_chunks"] == 0
    assert body["duplicate"] is False
    assert body["content_type"] == "text/plain"


def test_upload_rejects_unsupported_type(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("bad.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversized_files(client, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.integration.conftest import make_settings

    settings = make_settings(tmp_path).model_copy(update={"max_upload_size_mb": 1})
    app = create_app(settings)
    with TestClient(app) as c:
        big = "x" * (1 * 1024 * 1024 + 100)
        response = c.post(
            "/documents/upload",
            files={"file": ("big.txt", big.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 413


def test_duplicate_upload_returns_existing_document(client):
    first = upload_document(client, "notes.txt", SAMPLE)
    second = upload_document(client, "notes.txt", SAMPLE)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["duplicate"] is True
    assert second.json()["id"] == first.json()["id"]


def test_index_and_reindex(client):
    upload = upload_document(client, "notes.txt", SAMPLE)
    document_id = upload.json()["id"]

    index = client.post(f"/documents/{document_id}/index")
    assert index.status_code == 200
    body = index.json()
    assert body["status"] == "indexed"
    assert body["num_chunks"] > 0

    reindex = client.post(f"/documents/{document_id}/index")
    assert reindex.status_code == 200
    assert reindex.json()["num_chunks"] == body["num_chunks"]


def test_index_unknown_document_returns_404(client):
    assert client.post("/documents/nonexistent/index").status_code == 404


def test_list_and_get_documents(client):
    upload_document(client, "a.txt", SAMPLE)
    upload_document(client, "b.txt", SAMPLE + "Different content for b.")
    listing = client.get("/documents")
    assert listing.status_code == 200
    names = {doc["name"] for doc in listing.json()}
    assert {"a.txt", "b.txt"} <= names

    document_id = listing.json()[0]["id"]
    single = client.get(f"/documents/{document_id}")
    assert single.status_code == 200
    assert single.json()["id"] == document_id


def test_get_unknown_document_returns_404(client):
    assert client.get("/documents/missing").status_code == 404


def test_delete_document_removes_it(client):
    upload = upload_document(client, "gone.txt", SAMPLE)
    document_id = upload.json()["id"]
    client.post(f"/documents/{document_id}/index")

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 204
    assert client.get(f"/documents/{document_id}").status_code == 404
    assert client.post(f"/documents/{document_id}/index").status_code == 404


def test_index_empty_document_fails_gracefully(client):
    upload = upload_document(client, "empty.txt", "")
    assert upload.status_code == 201
    index = client.post(f"/documents/{upload.json()['id']}/index")
    assert index.status_code == 422
    assert client.get(f"/documents/{upload.json()['id']}").json()["status"] == "failed"
