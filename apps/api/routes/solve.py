"""POST /v1/solve: run real model inference on a real ARC task.

Errors are explicit and never faked:
- 404 if the task does not exist
- 400 if recursion_steps exceeds the configured maximum (or < 1)
- 503 if the requested model has no loadable checkpoint
Every solve attempt (success or failure) is logged to the ``runs`` table.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from apps.api.deps import SolveRequest, SolveResponse, index_dep, runtime_dep, settings_dep
from apps.worker.runtime import CheckpointUnavailableError, ModelRuntime
from packages.common.config import Settings
from packages.common.db import Run, session_scope
from packages.common.logging import get_logger
from packages.data.index import TaskIndex
from packages.inference.solver import result_to_dict

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["solve"])


def _new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


@router.post("/solve", response_model=SolveResponse)
def solve(
    req: SolveRequest,
    index: TaskIndex = Depends(index_dep),
    runtime: ModelRuntime = Depends(runtime_dep),
    settings: Settings = Depends(settings_dep),
) -> SolveResponse:
    model_id = req.model_id or settings.default_model_id
    run_id = _new_run_id()

    task = index.get(req.task_id)
    if task is None:
        _log_failed_run(run_id, req.task_id, model_id, req.recursion_steps, "task not found")
        raise HTTPException(status_code=404, detail=f"Task not found: {req.task_id}")

    if req.recursion_steps < 1 or req.recursion_steps > settings.max_recursion_steps:
        detail = (
            f"recursion_steps={req.recursion_steps} out of range "
            f"[1, {settings.max_recursion_steps}]"
        )
        _log_failed_run(run_id, req.task_id, model_id, req.recursion_steps, detail)
        raise HTTPException(status_code=400, detail=detail)

    if req.test_index < 0 or req.test_index >= task.num_test_pairs:
        raise HTTPException(status_code=400, detail=f"test_index out of range for {req.task_id}")

    try:
        result, loaded = runtime.solve_task(
            task,
            model_id=model_id,
            recursion_steps=req.recursion_steps,
            return_trace=req.return_trace,
            test_index=req.test_index,
        )
    except CheckpointUnavailableError as exc:
        _log_failed_run(run_id, req.task_id, model_id, req.recursion_steps, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = result_to_dict(result)
    dataset_hash = loaded.dataset_hash or index.dataset_hash

    _log_run(run_id, req.task_id, model_id, req.recursion_steps, result, loaded)

    return SolveResponse(
        run_id=run_id,
        task_id=req.task_id,
        model_id=model_id,
        recursion_steps=req.recursion_steps,
        prediction=payload["prediction"],
        trace=payload["trace"],
        exact_match=payload["exact_match"],
        pixel_accuracy=payload["pixel_accuracy"],
        shape_accuracy=payload["shape_accuracy"],
        latency_ms=payload["latency_ms"],
        checkpoint_hash=loaded.checkpoint_hash,
        dataset_hash=dataset_hash,
    )


def _log_run(run_id, task_id, model_id, steps, result, loaded) -> None:
    try:
        with session_scope() as session:
            session.add(
                Run(
                    id=run_id,
                    task_id=task_id,
                    model_id=model_id,
                    recursion_steps=steps,
                    prediction_json=result.prediction,
                    trace_json=result.trace,
                    exact_match=result.exact_match,
                    pixel_accuracy=result.pixel_accuracy,
                    shape_accuracy=result.shape_accuracy,
                    latency_ms=result.latency_ms,
                    error=None,
                )
            )
    except Exception:  # noqa: BLE001 - logging failure must not break inference
        logger.exception("Failed to persist run %s", run_id)


def _log_failed_run(run_id, task_id, model_id, steps, error) -> None:
    try:
        with session_scope() as session:
            session.add(
                Run(
                    id=run_id,
                    task_id=task_id,
                    model_id=model_id,
                    recursion_steps=steps,
                    error=error,
                )
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist failed run %s", run_id)
