"""Unit tests for API key parsing and tenant mapping."""

import pytest

from app.api.auth import parse_api_keys
from app.core.config import Settings


def test_parse_key_tenant_pairs():
    keys = parse_api_keys("sk-a:acme, sk-b:globex")
    assert keys == {"sk-a": "acme", "sk-b": "globex"}


def test_parse_bare_key_maps_to_default_tenant():
    keys = parse_api_keys("sk-shared", default_tenant="public")
    assert keys == {"sk-shared": "public"}


def test_parse_mixed_entries():
    keys = parse_api_keys("sk-a:acme,sk-b,  ,sk-c:globex", default_tenant="public")
    assert keys == {"sk-a": "acme", "sk-b": "public", "sk-c": "globex"}


def test_parse_empty_returns_no_keys():
    assert parse_api_keys("") == {}
    assert parse_api_keys("  ,  ") == {}


def test_auth_requires_keys_when_enabled():
    with pytest.raises(ValueError):
        Settings(auth_enabled=True, api_keys="")


def test_auth_can_be_disabled_without_keys():
    settings = Settings(auth_enabled=False, api_keys="")
    assert settings.auth_enabled is False
