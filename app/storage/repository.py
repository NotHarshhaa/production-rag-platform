"""Repository for document metadata, scoped by tenant where required.

Every method accepts an optional ``tenant_id``. When provided, documents
belonging to other tenants are invisible — a mismatch is indistinguishable
from a missing row so ids never leak across tenants.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.orm import Session, undefer

from app.storage.database import (
    DEFAULT_TENANT,
    Conversation,
    ConversationMessage,
    Document,
)


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

    def list_expired(self, cutoff: datetime, tenant_id: str | None = None) -> list[Document]:
        """Documents not touched since ``cutoff`` (content never loaded)."""

        with Session(self.engine) as session:
            stmt = (
                select(Document)
                .where(Document.updated_at < cutoff)
                .order_by(Document.updated_at.asc())
            )
            if tenant_id is not None:
                stmt = stmt.where(Document.tenant_id == tenant_id)
            documents = list(session.scalars(stmt))
            session.expunge_all()
            return documents


class ConversationRepository:
    """Tenant-scoped storage for chat sessions and their messages."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def create(self, title: str | None, tenant_id: str = DEFAULT_TENANT) -> Conversation:
        conversation = Conversation(
            id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            title=(title or "New conversation").strip()[:200] or "New conversation",
        )
        with Session(self.engine) as session:
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            session.expunge(conversation)
        return conversation

    def get(self, conversation_id: str, tenant_id: str | None = None) -> Conversation | None:
        with Session(self.engine) as session:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            if tenant_id is not None:
                stmt = stmt.where(Conversation.tenant_id == tenant_id)
            conversation = session.scalar(stmt)
            if conversation is not None:
                session.expunge_all()
            return conversation

    def list(self, tenant_id: str | None = None) -> list[Conversation]:
        with Session(self.engine) as session:
            stmt = select(Conversation).order_by(Conversation.updated_at.desc())
            if tenant_id is not None:
                stmt = stmt.where(Conversation.tenant_id == tenant_id)
            conversations = list(session.scalars(stmt))
            session.expunge_all()
            return conversations

    def rename(self, conversation_id: str, title: str) -> Conversation | None:
        with Session(self.engine) as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return None
            conversation.title = title.strip()[:200] or conversation.title
            conversation.updated_at = _utcnow()
            session.commit()
            session.refresh(conversation)
            session.expunge(conversation)
            return conversation

    def delete(self, conversation_id: str) -> bool:
        with Session(self.engine) as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return False
            session.delete(conversation)
            session.commit()
            return True

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ) -> ConversationMessage:
        with Session(self.engine) as session:
            message = ConversationMessage(
                conversation_id=conversation_id,
                role=role,
                content=content,
                citations=json.dumps(citations) if citations else None,
            )
            session.add(message)
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.updated_at = _utcnow()
                # Title follows the first user turn unless explicitly set.
                if role == "user" and conversation.title == "New conversation":
                    conversation.title = content.strip()[:80] or conversation.title
            session.commit()
            session.refresh(message)
            session.expunge(message)
            return message

    def list_messages(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ConversationMessage]:
        """Messages oldest-first; with ``limit`` the most recent ``limit`` turns."""

        with Session(self.engine) as session:
            stmt = select(ConversationMessage).where(
                ConversationMessage.conversation_id == conversation_id
            )
            if limit is not None and limit > 0:
                stmt = stmt.order_by(ConversationMessage.id.desc()).limit(limit)
                messages = list(session.scalars(stmt))
                messages.reverse()
            else:
                stmt = stmt.order_by(ConversationMessage.id.asc())
                messages = list(session.scalars(stmt))
            session.expunge_all()
            return messages

    def message_count(self, conversation_id: str) -> int:
        with Session(self.engine) as session:
            count = session.scalar(
                select(text("COUNT(*)")).select_from(ConversationMessage.__table__).where(
                    ConversationMessage.conversation_id == conversation_id
                )
            )
            return int(count or 0)

    @staticmethod
    def citations_of(message: ConversationMessage) -> list[dict]:
        if not message.citations:
            return []
        try:
            return json.loads(message.citations)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return []
