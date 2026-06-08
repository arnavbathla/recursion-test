"""Run retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import db_session
from packages.common.db import Run

router = APIRouter(prefix="/v1/runs", tags=["runs"])


def _serialize(run: Run) -> dict:
    return {
        "run_id": run.id,
        "task_id": run.task_id,
        "model_id": run.model_id,
        "recursion_steps": run.recursion_steps,
        "prediction": run.prediction_json,
        "trace": run.trace_json,
        "exact_match": run.exact_match,
        "pixel_accuracy": run.pixel_accuracy,
        "shape_accuracy": run.shape_accuracy,
        "latency_ms": run.latency_ms,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.get("")
def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(db_session),
) -> dict:
    rows = session.execute(select(Run).order_by(Run.created_at.desc()).limit(limit)).scalars().all()
    return {"runs": [_serialize(r) for r in rows]}


@router.get("/{run_id}")
def get_run(run_id: str, session: Session = Depends(db_session)) -> dict:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _serialize(run)
