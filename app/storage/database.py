"""SQLAlchemy engine setup and the document metadata model.

The same model serves SQLite (local development, tests) and PostgreSQL
(production) because the engine is built from ``DATABASE_URL``.
"""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.schema import ForeignKey, UniqueConstraint

DEFAULT_TENANT = "public"


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    """Metadata plus the raw bytes of an ingested document.

    ``content`` keeps the original file so re-indexing never needs the
    client to re-upload. It is deferred so listings never load it.
    ``tenant_id`` scopes documents to one tenant (see app/api/auth.py);
    duplicate detection is per tenant via the composite unique constraint.
    """

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("sha256", "tenant_id", name="uq_documents_sha_tenant"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    content_type: Mapped[str]
    size_bytes: Mapped[int]
    sha256: Mapped[str] = mapped_column(index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default=DEFAULT_TENANT, index=True)
    status: Mapped[str] = mapped_column(default="uploaded", index=True)
    num_chunks: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)
    content: Mapped[bytes] = mapped_column(deferred=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Document id={self.id!r} name={self.name!r} status={self.status!r}>"


class Conversation(Base):
    """A named chat session whose turns are persisted server-side.

    Conversations are scoped by ``tenant_id`` like documents; clients replay
    them by sending ``conversation_id`` on /query instead of resending the
    full message history.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default=DEFAULT_TENANT, index=True)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Conversation id={self.id!r} title={self.title!r}>"


class ConversationMessage(Base):
    """One turn in a conversation: role, content, and optional citations.

    ``citations`` holds the JSON array of citation objects for assistant
    turns so a client re-rendering a session sees the same sources.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str]
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ConversationMessage id={self.id!r} role={self.role!r}>"


def build_engine(database_url: str):
    """Create an engine, creating the parent directory for SQLite URLs."""

    if database_url.startswith("sqlite"):
        db_path = database_url.split("sqlite:///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )

    return create_engine(database_url, pool_pre_ping=True)


def init_db(engine) -> None:
    """Create tables and apply in-place upgrades for pre-tenant databases.

    Intentionally simple for the MVP; a migration tool (e.g. Alembic) should
    take over once the schema evolves beyond the initial release.
    """

    Base.metadata.create_all(engine)
    _ensure_tenant_column(engine)


def _ensure_tenant_column(engine) -> None:
    """Add ``tenant_id`` to databases created before multi-tenancy."""

    from sqlalchemy import inspect

    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("documents")}
    if "tenant_id" in columns:
        return
    with engine.begin() as connection:
        connection.execute(
            text(
                f"ALTER TABLE documents ADD COLUMN tenant_id VARCHAR(64) "
                f"NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
        )
    print("Upgraded documents table: added tenant_id column (default 'public')")
