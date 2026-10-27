"""Health and readiness endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app import __version__
from app.api.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["health"])


@router.get("/health")
def health(container: Annotated[ServiceContainer, Depends(get_container)]) -> dict:
    components: dict[str, str] = {}
    healthy = True

    try:
        components["database"] = "ok" if container.repository.ping() else "error"
    except Exception:
        components["database"] = "error"
    healthy = healthy and components["database"] == "ok"

    try:
        container.pipeline.vector_store.count_chunks()
        components["vector_store"] = "ok"
    except Exception:
        components["vector_store"] = "error"
    healthy = healthy and components["vector_store"] == "ok"

    payload = {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "environment": container.settings.app_env,
        "components": components,
    }
    if not healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=payload,
        )
    return payload
