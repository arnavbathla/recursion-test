"""FastAPI application entrypoint for the Recursive ARC Engine.

On startup: initialize the database (create_all), persist the model registry into
Postgres/SQLite, and warm the in-memory task index. The task index loads + validates
real ARC data; if the data has not been synced yet, startup logs a clear warning and
data-dependent endpoints will return explicit errors rather than fake data.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from packages.common.db import init_db
from packages.common.logging import configure_logging, get_logger
from packages.common.registry import sync_registry_to_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    try:
        n = sync_registry_to_db()
        logger.info("Synced %d model(s) from registry to DB", n)
    except Exception:  # noqa: BLE001
        logger.exception("Registry -> DB sync failed (continuing)")

    try:
        from packages.data.index import get_task_index, seed_tasks_to_db

        idx = get_task_index()
        logger.info("Warmed task index: %d tasks", len(idx.by_id))
        seed_tasks_to_db()
    except Exception:  # noqa: BLE001 - data may not be synced yet
        logger.warning(
            "Task index unavailable at startup. Run scripts/sync_arc_agi_2.sh. "
            "Data endpoints will return explicit errors until then."
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Recursive ARC Engine",
        version="0.1.0",
        description="TRM/HRM-inspired latent-recursive ARC solver served over a real API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from apps.api.routes import evaluate, health, models, runs, solve, tasks

    app.include_router(health.router)
    app.include_router(tasks.router)
    app.include_router(models.router)
    app.include_router(solve.router)
    app.include_router(runs.router)
    app.include_router(evaluate.router)
    return app


app = create_app()
