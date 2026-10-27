"""API key authentication with per-key tenant scoping.

Configure keys as comma-separated ``key:tenant`` pairs (a bare key maps to
``DEFAULT_TENANT``)::

    AUTH_ENABLED=true
    API_KEYS=sk-team-a:acme,sk-team-b:globex,sk-shared

Every tenant's documents and query results are isolated: retrieval filters
by ``tenant_id`` server-side, so one tenant's content can never appear in
another tenant's results. With AUTH_ENABLED=false all traffic is treated as
the default tenant (single-user mode).
"""

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings
from app.core.logging import get_logger
from app.storage.database import DEFAULT_TENANT

logger = get_logger(__name__)


def parse_api_keys(raw: str, default_tenant: str = DEFAULT_TENANT) -> dict[str, str]:
    """Parse ``key:tenant`` pairs (or bare keys) into a key -> tenant map."""

    keys: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, separator, tenant = entry.partition(":")
        if not separator or not tenant.strip():
            keys[key] = default_tenant
        else:
            keys[key] = tenant.strip()
    return keys


def _provided_key(request: Request) -> str | None:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip()
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def get_tenant(request: Request) -> str:
    """FastAPI dependency: authenticate the request and return its tenant."""

    container = request.app.state.container
    settings: Settings = container.settings
    if not settings.auth_enabled:
        return DEFAULT_TENANT

    provided = _provided_key(request)
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Send it as X-API-Key or Authorization: Bearer.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    expected = parse_api_keys(settings.api_keys, settings.default_tenant)
    for key, tenant in expected.items():
        if hmac.compare_digest(key, provided):
            return tenant

    logger.warning("Rejected request with unknown API key", extra={"path": request.url.path})
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


Tenant = Annotated[str, Depends(get_tenant)]
