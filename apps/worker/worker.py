"""RQ worker for asynchronous evaluation jobs.

The API enqueues ``process_eval_job`` onto the ``arc-eval`` queue. This worker runs
the real evaluation harness and writes results back into the ``eval_jobs`` table.

Start a worker:
    uv run python -m apps.worker.worker
"""

from __future__ import annotations

from typing import Any

from apps.worker.runtime import ModelRuntime
from packages.common.config import REPO_ROOT, get_settings
from packages.common.db import EvalJob, init_db, session_scope
from packages.common.logging import get_logger
from packages.eval.evaluate import run_evaluation
from packages.training.checkpoint import load_checkpoint

logger = get_logger(__name__)

QUEUE_NAME = "arc-eval"


def process_eval_job(job_id: str, model_id: str, split: str, depths: list[int]) -> dict[str, Any]:
    """Run an evaluation job and persist results. Safe to run in-process or via RQ."""
    init_db()
    runtime = ModelRuntime()
    try:
        loaded = runtime.load(model_id)
        payload = load_checkpoint(loaded.checkpoint_path)
        config = payload.get("config", {})
        out_path = REPO_ROOT / "reports" / f"{model_id}_{split}_eval.json"
        report = run_evaluation(
            loaded.checkpoint_path, config, split, depths, out_path=out_path, model_id=model_id
        )
        with session_scope() as session:
            row = session.get(EvalJob, job_id)
            if row is not None:
                row.metrics_json = report
                row.status = "completed"
        logger.info("Eval job %s completed", job_id)
        return report
    except Exception as exc:  # noqa: BLE001 - persist failure, then re-raise
        logger.exception("Eval job %s failed", job_id)
        with session_scope() as session:
            row = session.get(EvalJob, job_id)
            if row is not None:
                row.status = "failed"
                row.error = str(exc)
        raise


def main() -> int:
    from redis import Redis
    from rq import Queue, Worker

    settings = get_settings()
    conn = Redis.from_url(settings.redis_url)
    queue = Queue(QUEUE_NAME, connection=conn)
    logger.info("Starting RQ worker on queue=%s redis=%s", QUEUE_NAME, settings.redis_url)
    worker = Worker([queue], connection=conn)
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
