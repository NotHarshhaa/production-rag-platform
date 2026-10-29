"""Integration tests for document download and raw text extraction."""

import io


def test_download_document_attachment(client):
    content = b"This is a test document download content."
    upload = client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("export_test.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]

    response = client.get(f"/documents/{doc_id}/download")
    assert response.status_code == 200
    assert response.content == content
    disposition = response.headers.get("content-disposition", "")
    assert 'attachment; filename="export_test.txt"' in disposition


def test_download_document_non_existent_404(client):
    response = client.get("/documents/non-existent-id/download")
    assert response.status_code == 404


def test_get_raw_text(client):
    content = b"Line one.\n\nLine two."
    upload = client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("raw_test.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]

    response = client.get(f"/documents/{doc_id}/raw")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == doc_id
    assert data["name"] == "raw_test.txt"
    assert data["num_chunks"] > 0
    assert "Line one." in data["raw_text"]
