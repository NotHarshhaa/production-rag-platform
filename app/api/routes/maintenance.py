"""Operational maintenance endpoints, gated behind ADMIN_KEYS when auth is on."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_admin
from app.api.dependencies import ServiceContainer, get_container
from app.ingestion.retention import sweep_retention

router = APIRouter(
    prefix="/maintenance", tags=["maintenance"], dependencies=[Depends(require_admin)]
)

_Container = Annotated[ServiceContainer, Depends(get_container)]


@router.post("/retention")
def run_retention_sweep(container: _Container) -> dict:
    """Apply the retention policy immediately (RETENTION_DAYS, 0 disables)."""

    if container.settings.retention_days <= 0:
        raise HTTPException(
            status_code=400,
            detail="Retention is disabled. Set RETENTION_DAYS to a positive number of days.",
        )
    deleted = sweep_retention(container.pipeline, container.settings.retention_days)
    return {
        "retention_days": container.settings.retention_days,
        "deleted_count": len(deleted),
        "deleted_document_ids": deleted,
    }
