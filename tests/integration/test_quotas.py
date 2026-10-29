"""Per-key API quotas: key:tenant:quota parsing and 429 enforcement."""

from fastapi.testclient import TestClient

from app.api.auth import ApiKeyConfig, parse_api_key_configs, parse_api_keys
from app.main import create_app
from tests.integration.conftest import make_settings


def test_parse_api_key_configs_supports_quota_segment():
    configs = parse_api_key_configs("sk-a:acme:30,sk-b:globex,sk-c::5,sk-d:acme:notanumber")
    assert configs["sk-a"] == ApiKeyConfig(key="sk-a", tenant="acme", quota_per_minute=30)
    assert configs["sk-b"] == ApiKeyConfig(key="sk-b", tenant="globex", quota_per_minute=0)
    assert configs["sk-c"] == ApiKeyConfig(key="sk-c", tenant="public", quota_per_minute=5)
    # Non-numeric quota falls back to unlimited rather than failing startup.
    assert configs["sk-d"].quota_per_minute == 0


def test_parse_api_keys_stays_tenant_only():
    assert parse_api_keys("sk-a:acme:10,sk-b") == {"sk-a": "acme", "sk-b": "public"}


def test_quota_is_enforced_per_key(tmp_path):
    settings = make_settings(
        tmp_path, auth_enabled=True, api_keys="sk-limited:acme:3,sk-open:acme"
    )
    app = create_app(settings)
    with TestClient(app) as client:
        limited = [
            client.get("/documents", headers={"X-API-Key": "sk-limited"}).status_code
            for _ in range(5)
        ]
        assert limited == [200, 200, 200, 429, 429]

        # The other key on the same tenant is unaffected.
        open_codes = [
            client.get("/documents", headers={"X-API-Key": "sk-open"}).status_code
            for _ in range(5)
        ]
        assert open_codes == [200] * 5


def test_quota_rejection_reports_retry_after(tmp_path):
    settings = make_settings(tmp_path, auth_enabled=True, api_keys="sk-limited:acme:1")
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/documents", headers={"X-API-Key": "sk-limited"}).status_code == 200
        rejected = client.get("/documents", headers={"X-API-Key": "sk-limited"})
        assert rejected.status_code == 429
        assert rejected.headers.get("Retry-After") == "60"
        assert "quota" in rejected.json()["detail"].lower()
