"""Integration tests for the Web UI endpoint."""


def test_ui_root_endpoint_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Production RAG Platform" in response.text
    assert "Grounded Document Q&A" in response.text


def test_ui_alias_endpoint_returns_html(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Production RAG Platform" in response.text
