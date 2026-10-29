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
    documents = 0
    chunks = 0

    try:
        components["database"] = "ok" if container.repository.ping() else "error"
    except Exception:
        components["database"] = "error"
    healthy = healthy and components["database"] == "ok"

    try:
        chunks = container.pipeline.vector_store.count_chunks()
        components["vector_store"] = "ok"
    except Exception:
        components["vector_store"] = "error"
    healthy = healthy and components["vector_store"] == "ok"

    if components["database"] == "ok":
        try:
            documents = container.repository.count()
        except Exception:  # pragma: no cover - ping already succeeded
            documents = 0

    payload = {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "environment": container.settings.app_env,
        "components": components,
        "stats": {"documents": documents, "chunks": chunks},
    }
    if not healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=payload,
        )
    return payload
