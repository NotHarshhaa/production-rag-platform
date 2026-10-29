"""API key authentication with per-key tenant scoping, quotas, and admin keys.

Configure keys as comma-separated ``key:tenant[:quota]`` entries (a bare key
maps to ``DEFAULT_TENANT``)::

    AUTH_ENABLED=true
    API_KEYS=sk-team-a:acme,sk-team-b:globex:60,sk-shared
    ADMIN_KEYS=sk-team-a

The optional third segment is a per-key request quota per minute; ``0`` or
omission means unlimited. Quota rejections return 429 with ``Retry-After``.

Every tenant's documents and query results are isolated: retrieval filters
by ``tenant_id`` server-side, so one tenant's content can never appear in
another tenant's results. With AUTH_ENABLED=false all traffic is treated as
the default tenant (single-user mode). Maintenance endpoints additionally
require a key listed in ADMIN_KEYS when auth is enabled.
"""

import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.ratelimit import SlidingWindowLimiter
from app.observability.metrics import KEY_QUOTA_REJECTIONS
from app.storage.database import DEFAULT_TENANT

logger = get_logger(__name__)


@dataclass(frozen=True)
class ApiKeyConfig:
    """Resolved configuration for one API key."""

    key: str
    tenant: str
    quota_per_minute: int = 0  # 0 = unlimited


def parse_api_key_configs(
    raw: str, default_tenant: str = DEFAULT_TENANT
) -> dict[str, ApiKeyConfig]:
    """Parse ``key:tenant[:quota]`` entries into a key -> config map."""

    configs: dict[str, ApiKeyConfig] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split(":")]
        key = parts[0]
        if not key:
            continue
        tenant = parts[1] if len(parts) > 1 and parts[1] else default_tenant
        quota = 0
        if len(parts) > 2 and parts[2]:
            try:
                quota = max(0, int(parts[2]))
            except ValueError:
                logger.warning(
                    "Ignoring non-numeric quota in API_KEYS entry", extra={"entry": entry}
                )
        configs[key] = ApiKeyConfig(key=key, tenant=tenant, quota_per_minute=quota)
    return configs


def parse_api_keys(raw: str, default_tenant: str = DEFAULT_TENANT) -> dict[str, str]:
    """Parse ``key:tenant`` pairs (or bare keys) into a key -> tenant map."""

    return {
        key: config.tenant
        for key, config in parse_api_key_configs(raw, default_tenant).items()
    }


def _provided_key(request: Request) -> str | None:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip()
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def authenticate(request: Request) -> ApiKeyConfig:
    """Authenticate the request and enforce the key's quota, if configured."""

    container = request.app.state.container
    settings: Settings = container.settings
    if not settings.auth_enabled:
        return ApiKeyConfig(key="", tenant=DEFAULT_TENANT)

    provided = _provided_key(request)
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Send it as X-API-Key or Authorization: Bearer.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    expected = parse_api_key_configs(settings.api_keys, settings.default_tenant)
    config = next(
        (cfg for key, cfg in expected.items() if hmac.compare_digest(key, provided)), None
    )
    if config is None:
        logger.warning("Rejected request with unknown API key", extra={"path": request.url.path})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if config.quota_per_minute > 0:
        limiter = container.key_limiters.setdefault(
            config.quota_per_minute, SlidingWindowLimiter(config.quota_per_minute)
        )
        if not limiter.allow(config.key):
            KEY_QUOTA_REJECTIONS.inc()
            logger.warning(
                "API key quota exceeded", extra={"path": request.url.path, "tenant": config.tenant}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="API key quota exceeded. Try again later.",
                headers={"Retry-After": "60"},
            )
    return config


def get_tenant(request: Request) -> str:
    """FastAPI dependency: authenticate the request and return its tenant."""

    return authenticate(request).tenant


def require_admin(request: Request) -> None:
    """FastAPI dependency: gate maintenance endpoints behind ADMIN_KEYS."""

    settings: Settings = request.app.state.container.settings
    if not settings.auth_enabled:
        return
    config = authenticate(request)
    admin_keys = [k.strip() for k in settings.admin_keys.split(",") if k.strip()]
    if config.key and any(hmac.compare_digest(k, config.key) for k in admin_keys):
        return
    logger.warning("Rejected non-admin key on a maintenance endpoint")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="This endpoint requires an admin API key (ADMIN_KEYS).",
    )


Tenant = Annotated[str, Depends(get_tenant)]
