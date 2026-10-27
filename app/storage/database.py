"""SQLAlchemy engine setup and the document metadata model.

The same model serves SQLite (local development, tests) and PostgreSQL
(production) because the engine is built from ``DATABASE_URL``.
"""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    """Metadata plus the raw bytes of an ingested document.

    ``content`` keeps the original file so re-indexing never needs the
    client to re-upload. It is deferred so listings never load it.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    content_type: Mapped[str]
    size_bytes: Mapped[int]
    sha256: Mapped[str] = mapped_column(unique=True, index=True)
    status: Mapped[str] = mapped_column(default="uploaded", index=True)
    num_chunks: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)
    content: Mapped[bytes] = mapped_column(deferred=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Document id={self.id!r} name={self.name!r} status={self.status!r}>"


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
    """Create tables if they do not exist.

    Intentionally simple for the MVP; a migration tool (e.g. Alembic) should
    take over once the schema evolves beyond the initial release.
    """

    Base.metadata.create_all(engine)
