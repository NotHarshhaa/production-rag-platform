"""Repository for document metadata. Short-lived sessions keep connections clean."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.orm import Session, undefer

from app.storage.database import Document


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
    ) -> Document:
        document = Document(
            id=uuid.uuid4().hex,
            name=name,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            status="uploaded",
            content=content,
        )
        with Session(self.engine) as session:
            session.add(document)
            session.commit()
            session.refresh(document)
        return document

    def get(self, document_id: str) -> Document | None:
        with Session(self.engine) as session:
            stmt = (
                select(Document)
                .where(Document.id == document_id)
                .options(undefer(Document.content))
            )
            document = session.scalar(stmt)
            if document is not None:
                session.expunge_all()
            return document

    def get_by_sha256(self, sha256: str) -> Document | None:
        with Session(self.engine) as session:
            stmt = (
                select(Document)
                .where(Document.sha256 == sha256)
                .options(undefer(Document.content))
            )
            document = session.scalar(stmt)
            if document is not None:
                session.expunge_all()
            return document

    def list(self) -> list[Document]:
        with Session(self.engine) as session:
            stmt = select(Document).order_by(Document.created_at.desc())
            documents = list(session.scalars(stmt))
            session.expunge_all()
            return documents

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
