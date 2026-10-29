"""Document upload, indexing, update, listing, and deletion."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.auth import Tenant
from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import DocumentChunkOut, DocumentOut
from app.core.logging import get_logger
from app.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileTypeError,
    detect_content_type,
)
from app.ingestion.pipeline import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    IngestionError,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_UPLOAD_CHUNK = 1024 * 1024

_Container = Annotated[ServiceContainer, Depends(get_container)]


def _not_found(document_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document {document_id} not found",
    )


def _unsupported(exc: UnsupportedFileTypeError) -> HTTPException:
    allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=f"{exc} Allowed: {allowed}",
    )


def _read_upload(file: UploadFile, container: ServiceContainer) -> bytes:
    """Validate the file type and read it with the configured size cap."""

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
    return bytes(buffer)


def _to_dict(document) -> dict:
    return {
        "id": document.id,
        "name": document.name,
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "tenant_id": document.tenant_id,
        "status": document.status,
        "num_chunks": document.num_chunks,
        "error": document.error,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: Annotated[UploadFile, File(...)],
    container: _Container,
    tenant: Tenant,
    response: Response,
    auto_index: bool = False,
) -> DocumentOut:
    """Store a document.

    With ``auto_index=true`` the document is indexed immediately (or queued
    for the background workers when INGESTION_MODE=background, returning
    202); otherwise index it separately via POST /documents/{id}/index.
    """

    try:
        data = _read_upload(file, container)
        document_id, duplicate = container.pipeline.ingest(
            name=file.filename or "upload", data=data, tenant_id=tenant
        )
        document = container.repository.get(document_id, tenant)

        if auto_index and not duplicate:
            if container.background_ingestion:
                container.indexer.submit(document_id)
                response.status_code = status.HTTP_202_ACCEPTED
                return DocumentOut.model_validate(
                    {**_to_dict(document), "duplicate": duplicate}
                )
            container.pipeline.index_document(document_id, tenant_id=tenant)
            document = container.repository.get(document_id, tenant)

        return DocumentOut.model_validate({**_to_dict(document), "duplicate": duplicate})
    except HTTPException:
        raise
    except UnsupportedFileTypeError as exc:
        raise _unsupported(exc) from exc
    except Exception as exc:
        logger.exception("Upload failed", extra={"file": file.filename})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store the uploaded document",
        ) from exc


@router.post("/{document_id}/index", response_model=DocumentOut)
def index_document(
    document_id: str, container: _Container, tenant: Tenant, response: Response
) -> DocumentOut:
    """Index or re-index a document.

    Synchronous by default. With INGESTION_MODE=background the job is queued
    and this endpoint returns 202 immediately; poll GET /documents/{id}.
    """

    document = container.repository.get(document_id, tenant)
    if document is None:
        raise _not_found(document_id)

    if container.background_ingestion:
        container.indexer.submit(document_id)
        response.status_code = status.HTTP_202_ACCEPTED
        return DocumentOut.model_validate(_to_dict(document))

    try:
        container.pipeline.index_document(document_id, tenant_id=tenant)
    except IngestionError as exc:
        failed = container.repository.get(document_id, tenant)
        if failed is not None and failed.status == "failed":
            raise HTTPException(status_code=422, detail=failed.error or str(exc)) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    document = container.repository.get(document_id, tenant)
    return DocumentOut.model_validate(_to_dict(document))


@router.put("/{document_id}", response_model=DocumentOut)
def replace_document(
    document_id: str,
    file: Annotated[UploadFile, File(...)],
    container: _Container,
    tenant: Tenant,
) -> DocumentOut:
    """Replace a document's content and re-index it."""

    try:
        data = _read_upload(file, container)
        container.pipeline.replace_document(
            document_id, name=file.filename or document_id, data=data, tenant_id=tenant
        )
    except HTTPException:
        raise
    except DocumentNotFoundError as exc:
        raise _not_found(document_id) from exc
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise _unsupported(exc) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    document = container.repository.get(document_id, tenant)
    return DocumentOut.model_validate(_to_dict(document))


@router.get("", response_model=list[DocumentOut])
def list_documents(container: _Container, tenant: Tenant) -> list[DocumentOut]:
    return [
        DocumentOut.model_validate(_to_dict(d)) for d in container.repository.list(tenant)
    ]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, container: _Container, tenant: Tenant) -> DocumentOut:
    document = container.repository.get(document_id, tenant)
    if document is None:
        raise _not_found(document_id)
    return DocumentOut.model_validate(_to_dict(document))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, container: _Container, tenant: Tenant) -> None:
    try:
        container.pipeline.delete_document(document_id, tenant_id=tenant)
    except DocumentNotFoundError as exc:
        raise _not_found(document_id) from exc
@router.get("/{document_id}/chunks", response_model=list[DocumentChunkOut])
def get_document_chunks(
    document_id: str, container: _Container, tenant: Tenant
) -> list[DocumentChunkOut]:
    """Retrieve all indexed chunks for a document, ordered by page and index."""
    document = container.repository.get(document_id, tenant)
    if document is None:
        raise _not_found(document_id)

    chunks: list[DocumentChunkOut] = []
    for record in container.pipeline.vector_store.iter_chunks(document_id):
        meta = record.metadata
        if meta.get("tenant_id") == tenant:
            text = meta.get("text", "")
            chunks.append(
                DocumentChunkOut(
                    chunk_id=meta.get("chunk_id", record.id),
                    document_id=document_id,
                    chunk_index=meta.get("chunk_index", 0),
                    page=meta.get("page", 1),
                    text=text,
                    char_count=len(text),
                    tenant_id=tenant,
                )
            )
    chunks.sort(key=lambda c: (c.page, c.chunk_index))
    return chunks

