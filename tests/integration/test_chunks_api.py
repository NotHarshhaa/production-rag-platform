"""Integration tests for the document chunks endpoint."""

import io


def test_get_document_chunks_returns_ordered_chunks(client):
    file_content = b"# Architecture Overview\n\nThis is paragraph one.\n\nThis is paragraph two."
    upload = client.post(
        "/documents/upload?auto_index=true",
        files={"file": ("arch.md", io.BytesIO(file_content), "text/markdown")},
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]

    response = client.get(f"/documents/{doc_id}/chunks")
    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) > 0

    first = chunks[0]
    assert first["document_id"] == doc_id
    assert "chunk_id" in first
    assert "page" in first
    assert "chunk_index" in first
    assert "text" in first
    assert first["char_count"] == len(first["text"])


def test_get_document_chunks_non_existent_404(client):
    response = client.get("/documents/non-existent-uuid/chunks")
    assert response.status_code == 404
