"""Retention policy: expired-document sweeps and the maintenance endpoint."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingestion.retention import RetentionSweeper, sweep_retention
from app.main import create_app
from tests.integration.conftest import make_settings


def _backdate_document(database_url: str, document_id: str, days: int) -> None:
    path = database_url.split("sqlite:///", 1)[-1]
    with sqlite3.connect(path) as connection:
        old = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        connection.execute(
            "UPDATE documents SET updated_at = ? WHERE id = ?", (old, document_id)
        )


def test_sweep_retention_deletes_only_expired_documents(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        fresh = client.post(
            "/documents/upload",
            files={"file": ("fresh.md", b"fresh content " * 10, "text/plain")},
        ).json()
        stale = client.post(
            "/documents/upload",
            files={"file": ("stale.md", b"stale content " * 10, "text/plain")},
        ).json()
        _backdate_document(settings.database_url, stale["id"], days=30)

        pipeline = app.state.container.pipeline
        deleted = sweep_retention(pipeline, retention_days=7)

        assert deleted == [stale["id"]]
        remaining = {d["id"] for d in client.get("/documents").json()}
        assert remaining == {fresh["id"]}


def test_sweep_retention_disabled_is_a_noop(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        client.post(
            "/documents/upload",
            files={"file": ("doc.md", b"content " * 10, "text/plain")},
        )
        assert sweep_retention(app.state.container.pipeline, retention_days=0) == []
        assert len(client.get("/documents").json()) == 1


def test_maintenance_endpoint_requires_retention_enabled(tmp_path):
    app = create_app(make_settings(tmp_path))  # retention_days defaults to 0
    with TestClient(app) as client:
        response = client.post("/maintenance/retention")
        assert response.status_code == 400
        assert "RETENTION_DAYS" in response.json()["detail"]


def test_maintenance_endpoint_sweeps_on_demand(tmp_path):
    settings = make_settings(tmp_path, retention_days=7)
    app = create_app(settings)
    with TestClient(app) as client:
        stale = client.post(
            "/documents/upload",
            files={"file": ("old.md", b"old content " * 10, "text/plain")},
        ).json()
        _backdate_document(settings.database_url, stale["id"], days=30)

        response = client.post("/maintenance/retention")
        assert response.status_code == 200
        body = response.json()
        assert body["deleted_count"] == 1
        assert body["deleted_document_ids"] == [stale["id"]]
        assert body["retention_days"] == 7


def test_maintenance_endpoint_requires_admin_key_when_auth_enabled(tmp_path):
    settings = make_settings(
        tmp_path,
        retention_days=7,
        auth_enabled=True,
        api_keys="sk-member:acme,sk-admin:acme",
        admin_keys="sk-admin",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        member = client.post("/maintenance/retention", headers={"X-API-Key": "sk-member"})
        assert member.status_code == 403

        admin = client.post("/maintenance/retention", headers={"X-API-Key": "sk-admin"})
        assert admin.status_code == 200


def test_retention_sweeper_thread_stops_cleanly(tmp_path):
    settings = make_settings(tmp_path, retention_days=7)
    app = create_app(settings)
    with TestClient(app):
        sweeper: RetentionSweeper = app.state.container.retention_sweeper
        assert sweeper is not None
        sweeper.start()
        sweeper.start()  # starting twice is a no-op
        sweeper.stop()
        sweeper.stop()  # stopping twice is a no-op


def test_settings_reject_invalid_retention_values():
    with pytest.raises(ValueError):
        Settings(retention_days=-1)
    with pytest.raises(ValueError):
        Settings(retention_sweep_interval_seconds=10)
