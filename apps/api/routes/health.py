"""Health + readiness routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.deps import index_dep, runtime_dep
from apps.worker.runtime import ModelRuntime
from packages.data.index import TaskIndex

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@router.get("/readyz")
def readyz(index: TaskIndex = Depends(index_dep), runtime: ModelRuntime = Depends(runtime_dep)) -> dict:
    return {
        "ok": True,
        "num_tasks": len(index.by_id),
        "dataset_hash": index.dataset_hash,
        "runtime": runtime.info(),
    }
