"""Shared FastAPI dependencies and Pydantic request/response schemas."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.worker.runtime import ModelRuntime, get_runtime
from packages.common.config import Settings, get_settings
from packages.common.db import get_session_factory
from packages.data.index import TaskIndex, get_task_index


def settings_dep() -> Settings:
    return get_settings()


def index_dep() -> TaskIndex:
    return get_task_index()


def runtime_dep() -> ModelRuntime:
    return get_runtime()


def db_session() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


# ---------- Request / response models ----------

Grid = list[list[int]]


class SolveRequest(BaseModel):
    task_id: str
    model_id: str | None = None
    recursion_steps: int = 32
    return_trace: bool = True
    test_index: int = 0


class TraceItem(BaseModel):
    step: int
    height: int
    width: int
    grid: Grid


class SolveResponse(BaseModel):
    run_id: str
    task_id: str
    model_id: str
    recursion_steps: int
    prediction: Grid
    trace: list[TraceItem]
    exact_match: bool | None
    pixel_accuracy: float | None
    shape_accuracy: bool | None
    latency_ms: float
    checkpoint_hash: str
    dataset_hash: str | None


class EvaluateRequest(BaseModel):
    model_id: str | None = None
    split: str = "train_holdout"
    depths: list[int] = [1, 2, 4, 8, 16, 32]
    limit: int | None = None
    async_job: bool = False


class TaskSummary(BaseModel):
    task_id: str
    split: str
    num_train_pairs: int
    num_test_pairs: int
