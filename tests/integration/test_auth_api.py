"""Integration tests for API-key authentication and tenant isolation."""

import pytest

from tests.integration.conftest import upload_as


def _auth_headers(key: str) -> dict:
    return {"X-API-Key": key}


def test_missing_key_is_unauthorized(auth_client):
    response = upload_as(auth_client, "a.txt", "hello")
    assert response.status_code == 401
    assert auth_client.get("/documents").status_code == 401
    assert (
        auth_client.post("/query", json={"question": "hi"}).status_code == 401
    )


def test_invalid_key_is_unauthorized(auth_client):
    response = upload_as(auth_client, "a.txt", "hello", _auth_headers("sk-wrong"))
    assert response.status_code == 401


def test_bearer_authorization_is_supported(auth_client):
    response = upload_as(
        auth_client, "a.txt", "Bearer auth works", {"Authorization": "Bearer sk-acme"}
    )
    assert response.status_code == 201
    assert response.json()["tenant_id"] == "acme"


def test_health_and_metrics_stay_open(auth_client):
    assert auth_client.get("/health").status_code == 200
    assert auth_client.get("/metrics").status_code == 200


def test_documents_are_scoped_per_tenant(auth_client):
    acme = upload_as(
        auth_client, "acme.txt", "Acme confidential plans.", _auth_headers("sk-acme")
    )
    globex = upload_as(
        auth_client, "globex.txt", "Globex secret roadmap.", _auth_headers("sk-globex")
    )
    assert acme.status_code == 201 and globex.status_code == 201

    acme_docs = auth_client.get("/documents", headers=_auth_headers("sk-acme")).json()
    globex_docs = auth_client.get("/documents", headers=_auth_headers("sk-globex")).json()
    assert {doc["name"] for doc in acme_docs} == {"acme.txt"}
    assert {doc["name"] for doc in globex_docs} == {"globex.txt"}


def test_cannot_read_or_delete_other_tenant_document(auth_client):
    acme_id = upload_as(
        auth_client, "acme.txt", "Acme confidential plans.", _auth_headers("sk-acme")
    ).json()["id"]
    globex = _auth_headers("sk-globex")
    acme = _auth_headers("sk-acme")

    assert auth_client.get(f"/documents/{acme_id}", headers=globex).status_code == 404
    assert auth_client.post(f"/documents/{acme_id}/index", headers=globex).status_code == 404
    assert auth_client.delete(f"/documents/{acme_id}", headers=globex).status_code == 404
    # Owner can still work with it.
    assert auth_client.get(f"/documents/{acme_id}", headers=acme).status_code == 200


def test_queries_only_retrieve_own_tenant_documents(auth_client):
    for key, name, body in (
        ("sk-acme", "acme-notes.txt", "The Acme widget uses a flux capacitor."),
        ("sk-globex", "globex-notes.txt", "The Globex device uses a warp core."),
    ):
        upload = upload_as(auth_client, name, body * 4, _auth_headers(key))
        assert upload.status_code == 201
        document_id = upload.json()["id"]
        indexed = auth_client.post(
            f"/documents/{document_id}/index", headers=_auth_headers(key)
        )
        assert indexed.status_code == 200

    acme_answer = auth_client.post(
        "/query",
        json={"question": "What does the Acme widget use?"},
        headers=_auth_headers("sk-acme"),
    )
    globex_answer = auth_client.post(
        "/query",
        json={"question": "What does the Globex device use?"},
        headers=_auth_headers("sk-globex"),
    )
    assert acme_answer.status_code == 200 and globex_answer.status_code == 200

    acme_sources = {chunk["document_name"] for chunk in acme_answer.json()["retrieval"]["chunks"]}
    globex_sources = {
        chunk["document_name"] for chunk in globex_answer.json()["retrieval"]["chunks"]
    }
    assert acme_sources == {"acme-notes.txt"}
    assert globex_sources == {"globex-notes.txt"}


def test_duplicate_detection_is_per_tenant(auth_client):
    content = "Identical content across tenants."
    first = upload_as(auth_client, "a.txt", content, _auth_headers("sk-acme"))
    second = upload_as(auth_client, "a.txt", content, _auth_headers("sk-globex"))
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is False
    assert first.json()["id"] != second.json()["id"]

    again = upload_as(auth_client, "a.txt", content, _auth_headers("sk-acme"))
    assert again.json()["duplicate"] is True
    assert again.json()["id"] == first.json()["id"]


def test_auth_disabled_by_default(client):
    response = upload_as(client, "a.txt", "plain upload")
    assert response.status_code == 201
    assert response.json()["tenant_id"] == "public"


@pytest.mark.parametrize("header", ["X-API-Key", "Authorization"])
def test_empty_key_header_is_unauthorized(auth_client, header):
    headers = {header: "Bearer " if header == "Authorization" else ""}
    response = auth_client.post("/query", json={"question": "hi"}, headers=headers)
    assert response.status_code == 401
