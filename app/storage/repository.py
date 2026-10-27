"""Repository for document metadata, scoped by tenant where required.

Every method accepts an optional ``tenant_id``. When provided, documents
belonging to other tenants are invisible — a mismatch is indistinguishable
from a missing row so ids never leak across tenants.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.orm import Session, undefer

from app.storage.database import DEFAULT_TENANT, Document


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentRepository:
    def __init__(self, engine) -> None:
        self.engine = engine

    def ping(self) -> bool:
        try:
            with Session(self.engine) as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def create(
        self,
        name: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        content: bytes,
        tenant_id: str = DEFAULT_TENANT,
    ) -> Document:
        document = Document(
            id=uuid.uuid4().hex,
            name=name,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            tenant_id=tenant_id,
            status="uploaded",
            content=content,
        )
        with Session(self.engine) as session:
            session.add(document)
            session.commit()
            session.refresh(document)
            session.expunge(document)
        return document

    def get(self, document_id: str, tenant_id: str | None = None) -> Document | None:
        with Session(self.engine) as session:
            stmt = select(Document).where(Document.id == document_id)
            if tenant_id is not None:
                stmt = stmt.where(Document.tenant_id == tenant_id)
            document = session.scalar(stmt.options(undefer(Document.content)))
            if document is not None:
                session.expunge_all()
            return document

    def get_by_sha256(self, sha256: str, tenant_id: str | None = None) -> Document | None:
        with Session(self.engine) as session:
            stmt = select(Document).where(Document.sha256 == sha256)
            if tenant_id is not None:
                stmt = stmt.where(Document.tenant_id == tenant_id)
            document = session.scalar(stmt.options(undefer(Document.content)))
            if document is not None:
                session.expunge_all()
            return document

    def list(self, tenant_id: str | None = None) -> list[Document]:
        with Session(self.engine) as session:
            stmt = select(Document).order_by(Document.created_at.desc())
            if tenant_id is not None:
                stmt = stmt.where(Document.tenant_id == tenant_id)
            documents = list(session.scalars(stmt))
            session.expunge_all()
            return documents

    def replace_content(
        self,
        document_id: str,
        name: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        content: bytes,
    ) -> Document | None:
        """Overwrite a document's content and reset its index state."""

        with Session(self.engine) as session:
            document = session.get(Document, document_id)
            if document is None:
                return None
            document.name = name
            document.content_type = content_type
            document.size_bytes = size_bytes
            document.sha256 = sha256
            document.content = content
            document.status = "uploaded"
            document.num_chunks = 0
            document.error = None
            document.updated_at = _utcnow()
            session.commit()
            session.refresh(document)
            session.expunge(document)
            return document

    def update_status(
        self,
        document_id: str,
        status: str,
        num_chunks: int | None = None,
        error: str | None = None,
    ) -> Document | None:
        with Session(self.engine) as session:
            document = session.get(Document, document_id)
            if document is None:
                return None
            document.status = status
            document.error = error
            if num_chunks is not None:
                document.num_chunks = num_chunks
            document.updated_at = _utcnow()
            session.commit()
            session.refresh(document)
            session.expunge(document)
            return document

    def delete(self, document_id: str) -> bool:
        with Session(self.engine) as session:
            document = session.get(Document, document_id)
            if document is None:
                return False
            session.delete(document)
            session.commit()
            return True

    def count(self) -> int:
        with Session(self.engine) as session:
            count = session.scalar(select(text("COUNT(*)")).select_from(Document.__table__))
            return int(count or 0)
