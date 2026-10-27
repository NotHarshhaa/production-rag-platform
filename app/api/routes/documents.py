"""Document upload, indexing, listing, and deletion."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import DocumentOut
from app.core.logging import get_logger
from app.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileTypeError,
    detect_content_type,
)
from app.ingestion.pipeline import DocumentNotFoundError, IngestionError

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_UPLOAD_CHUNK = 1024 * 1024

_Container = Annotated[ServiceContainer, Depends(get_container)]


def _not_found(document_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document {document_id} not found",
    )


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: Annotated[UploadFile, File(...)],
    container: _Container,
) -> DocumentOut:
    """Store a document. Index it separately via POST /documents/{id}/index."""

    try:
        # Validate the file type before reading the body into memory.
        detect_content_type(file.filename or "")

        limit = container.settings.max_upload_size_bytes
        buffer = bytearray()
        while True:
            chunk = file.file.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File exceeds the {container.settings.max_upload_size_mb} MB upload limit"
                    ),
                )

        document_id, duplicate = container.pipeline.ingest(
            name=file.filename or "upload", data=bytes(buffer)
        )
        document = container.repository.get(document_id)
        return DocumentOut.model_validate({**_to_dict(document), "duplicate": duplicate})
    except HTTPException:
        raise
    except UnsupportedFileTypeError as exc:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{exc} Allowed: {allowed}",
        ) from exc
    except Exception as exc:
        logger.exception("Upload failed", extra={"file": file.filename})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store the uploaded document",
        ) from exc


@router.post("/{document_id}/index", response_model=DocumentOut)
def index_document(document_id: str, container: _Container) -> DocumentOut:
    """Index or re-index a stored document (idempotent)."""

    try:
        container.pipeline.index_document(document_id)
    except DocumentNotFoundError as exc:
        raise _not_found(document_id) from exc
    except IngestionError as exc:
        document = container.repository.get(document_id)
        if document is not None and document.status == "failed":
            raise HTTPException(
                status_code=422,
                detail=document.error or str(exc),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    document = container.repository.get(document_id)
    return DocumentOut.model_validate(_to_dict(document))


@router.get("", response_model=list[DocumentOut])
def list_documents(container: _Container) -> list[DocumentOut]:
    return [DocumentOut.model_validate(_to_dict(d)) for d in container.repository.list()]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, container: _Container) -> DocumentOut:
    document = container.repository.get(document_id)
    if document is None:
        raise _not_found(document_id)
    return DocumentOut.model_validate(_to_dict(document))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, container: _Container) -> None:
    try:
        container.pipeline.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise _not_found(document_id) from exc


def _to_dict(document) -> dict:
    return {
        "id": document.id,
        "name": document.name,
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "status": document.status,
        "num_chunks": document.num_chunks,
        "error": document.error,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }
