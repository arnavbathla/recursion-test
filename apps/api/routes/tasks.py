"""Task listing + retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from apps.api.deps import TaskSummary, index_dep
from packages.data.index import TaskIndex

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

VALID_SPLITS = {"training", "evaluation", "train", "validation", "train_holdout"}


@router.get("")
def list_tasks(
    split: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    index: TaskIndex = Depends(index_dep),
) -> dict:
    if split is not None and split not in VALID_SPLITS:
        raise HTTPException(status_code=400, detail=f"Invalid split: {split}")
    tasks = index.list_split(split)
    total = len(tasks)
    page = tasks[offset : offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "split": split,
        "tasks": [
            TaskSummary(
                task_id=t.task_id,
                split=t.split,
                num_train_pairs=t.num_train_pairs,
                num_test_pairs=t.num_test_pairs,
            ).model_dump()
            for t in page
        ],
    }


@router.get("/{task_id}")
def get_task(task_id: str, index: TaskIndex = Depends(index_dep)) -> dict:
    task = index.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return {
        "task_id": task.task_id,
        "split": task.split,
        "num_train_pairs": task.num_train_pairs,
        "num_test_pairs": task.num_test_pairs,
        "train": [{"input": p.input, "output": p.output} for p in task.train],
        "test": [
            {"input": p.input, "output": p.output} for p in task.test
        ],
    }
