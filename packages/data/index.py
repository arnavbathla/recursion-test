"""In-memory task index + optional DB seeding.

The index loads + validates all ARC tasks once and exposes lookup by id and by
logical split (train / validation / training / evaluation). It also computes the
dataset hash so callers can attach provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from packages.common.config import get_settings
from packages.common.hashing import dict_sha256
from packages.common.logging import get_logger
from packages.data.ingest_arc import DATASET_NAME, ingest_arc
from packages.data.schema import ARCTask
from packages.data.splits import select_split

logger = get_logger(__name__)


@dataclass
class TaskIndex:
    by_id: dict[str, ARCTask]
    split_members: dict[str, set[str]]
    dataset_hash: str

    def get(self, task_id: str) -> ARCTask | None:
        return self.by_id.get(task_id)

    def list_split(self, split: str | None) -> list[ARCTask]:
        if split is None:
            return sorted(self.by_id.values(), key=lambda t: t.task_id)
        ids = self.split_members.get(split, set())
        return sorted((self.by_id[i] for i in ids), key=lambda t: t.task_id)

    def split_of(self, task_id: str) -> str:
        return self.by_id[task_id].split if task_id in self.by_id else "unknown"


@lru_cache
def get_task_index() -> TaskIndex:
    root = get_settings().arc_data_root
    tasks, manifest = ingest_arc(root)
    by_id = {t.task_id: t for t in tasks}

    members: dict[str, set[str]] = {
        "training": {t.task_id for t in tasks if t.split == "training"},
        "evaluation": {t.task_id for t in tasks if t.split == "evaluation"},
    }
    # Logical splits derived from the official training split.
    for logical in ("train", "validation"):
        members[logical] = {
            t.task_id
            for t in select_split(tasks, logical, for_training=False)
        }
    members["train_holdout"] = set(members["validation"])

    dataset_hash = manifest.get("dataset_hash") or dict_sha256(sorted(by_id))
    logger.info("TaskIndex loaded: %d tasks (hash=%s)", len(by_id), dataset_hash[:12])
    return TaskIndex(by_id=by_id, split_members=members, dataset_hash=dataset_hash)


def seed_tasks_to_db(limit: int | None = None) -> int:
    """Insert/refresh task rows in the DB. Returns the number of tasks seeded."""
    from packages.common.db import Task, init_db, session_scope

    init_db()
    index = get_task_index()
    tasks = list(index.by_id.values())
    if limit:
        tasks = tasks[:limit]
    n = 0
    with session_scope() as session:
        for t in tasks:
            existing = session.get(Task, t.task_id)
            raw = t.to_raw()
            if existing is None:
                session.add(
                    Task(
                        id=t.task_id,
                        split=t.split,
                        raw_json=raw,
                        dataset_name=DATASET_NAME,
                        dataset_hash=index.dataset_hash,
                    )
                )
            else:
                existing.split = t.split
                existing.raw_json = raw
                existing.dataset_hash = index.dataset_hash
            n += 1
    logger.info("Seeded %d tasks into DB", n)
    return n
