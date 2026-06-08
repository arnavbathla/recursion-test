"""Database engine, session, and ORM models.

Uses ``DATABASE_URL`` from settings. Defaults to a local SQLite file so tests and
local runs work with zero infrastructure; Docker Compose points this at Postgres.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from packages.common.config import get_settings
from packages.common.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    split: Mapped[str] = mapped_column(String(32), index=True)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    dataset_name: Mapped[str] = mapped_column(String(128))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    checkpoint_path: Mapped[str] = mapped_column(String(512))
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), index=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    recursion_steps: Mapped[int] = mapped_column(Integer)
    prediction_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    trace_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    exact_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pixel_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    shape_accuracy: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class EvalJob(Base):
    __tablename__ = "eval_jobs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    split: Mapped[str] = mapped_column(String(32))
    depths_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, connect_args=connect_args)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, class_=Session)
        logger.info("Database engine created: %s", url.split("@")[-1])
    return _engine


def init_db() -> None:
    """Create all tables if they do not exist (idempotent)."""
    get_engine()
    Base.metadata.create_all(_engine)


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
