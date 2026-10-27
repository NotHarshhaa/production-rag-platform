"""Integration tests for the health endpoint."""


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["database"] == "ok"
    assert body["components"]["vector_store"] == "ok"
    assert body["version"]


def test_metrics_endpoint_exposes_counters(client):
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_http_requests_total" in response.text
    assert "rag_http_request_latency_seconds" in response.text


def test_request_id_header_present(client):
    response = client.get("/health")
    assert response.headers.get("x-request-id")
