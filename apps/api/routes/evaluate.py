"""Evaluation routes: synchronous small evals or enqueued async jobs."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import EvaluateRequest, db_session, runtime_dep, settings_dep
from apps.worker.runtime import CheckpointUnavailableError, ModelRuntime
from packages.common.config import Settings
from packages.common.db import EvalJob, session_scope
from packages.common.logging import get_logger
from packages.eval.evaluate import run_evaluation
from packages.training.checkpoint import load_checkpoint

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["evaluate"])


@router.post("/evaluate")
def evaluate(
    req: EvaluateRequest,
    runtime: ModelRuntime = Depends(runtime_dep),
    settings: Settings = Depends(settings_dep),
) -> dict:
    model_id = req.model_id or settings.default_model_id
    for d in req.depths:
        if d < 1 or d > settings.max_recursion_steps:
            raise HTTPException(status_code=400, detail=f"depth {d} out of range")

    try:
        loaded = runtime.load(model_id)
    except CheckpointUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    job_id = f"eval_{uuid.uuid4().hex[:16]}"
    with session_scope() as session:
        session.add(
            EvalJob(
                id=job_id, model_id=model_id, split=req.split,
                depths_json=req.depths, status="pending",
            )
        )

    if req.async_job:
        try:
            from redis import Redis
            from rq import Queue

            from apps.worker.worker import QUEUE_NAME, process_eval_job

            conn = Redis.from_url(settings.redis_url)
            conn.ping()
            queue = Queue(QUEUE_NAME, connection=conn)
            queue.enqueue(process_eval_job, job_id, model_id, req.split, req.depths, job_timeout=3600)
            with session_scope() as session:
                row = session.get(EvalJob, job_id)
                if row:
                    row.status = "queued"
            return {"eval_job_id": job_id, "status": "queued", "async": True}
        except Exception as exc:  # noqa: BLE001 - redis unavailable -> explicit error
            with session_scope() as session:
                row = session.get(EvalJob, job_id)
                if row:
                    row.status = "failed"
                    row.error = f"enqueue failed: {exc}"
            raise HTTPException(
                status_code=503,
                detail=f"Async evaluation requested but queue is unavailable: {exc}",
            ) from exc

    # Synchronous path (bounded by ``limit`` to keep request latency sane).
    payload = load_checkpoint(loaded.checkpoint_path)
    config = payload.get("config", {})
    report = run_evaluation(
        loaded.checkpoint_path, config, req.split, req.depths,
        out_path=None, limit=req.limit, model_id=model_id,
    )
    with session_scope() as session:
        row = session.get(EvalJob, job_id)
        if row:
            row.status = "completed"
            row.metrics_json = report
    return {"eval_job_id": job_id, "status": "completed", "async": False, "report": report}


@router.get("/evals")
def list_evals(session: Session = Depends(db_session)) -> dict:
    rows = session.execute(select(EvalJob).order_by(EvalJob.created_at.desc()).limit(100)).scalars().all()
    return {
        "evals": [
            {
                "eval_job_id": e.id,
                "model_id": e.model_id,
                "split": e.split,
                "depths": e.depths_json,
                "status": e.status,
                "metrics": e.metrics_json,
                "error": e.error,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in rows
        ]
    }
