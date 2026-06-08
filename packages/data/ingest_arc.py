"""Ingest official ARC-AGI-2 JSON files into validated ARCTask objects + a manifest.

Usage:
    python -m packages.data.ingest_arc data/raw/ARC-AGI-2
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from packages.common.config import REPO_ROOT
from packages.common.hashing import dict_sha256, file_sha256, git_commit
from packages.common.logging import get_logger
from packages.common.storage import write_json
from packages.data.schema import ARCPair, ARCTask

logger = get_logger(__name__)

DATASET_NAME = "ARC-AGI-2"
MANIFEST_PATH = "artifacts/arc_agi_2_manifest.json"


def load_task(path: Path, split: str) -> ARCTask:
    """Load and validate a single ARC task JSON file."""
    if split not in ("training", "evaluation"):
        raise ValueError(f"split must be 'training' or 'evaluation', got {split!r}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "train" not in raw or "test" not in raw:
        raise ValueError(f"{path} is not a valid ARC task (missing train/test)")
    task = ARCTask(
        task_id=path.stem,
        split=split,  # type: ignore[arg-type]
        train=[ARCPair(**p) for p in raw["train"]],
        test=[ARCPair(**p) for p in raw["test"]],
    )
    return task


def _split_dir(root: Path, split: str) -> Path:
    return root / "data" / split


def ingest_arc(root: str | Path) -> tuple[list[ARCTask], dict]:
    """Load all training + evaluation tasks under ``root`` and build a manifest dict."""
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = REPO_ROOT / root_path
    if not root_path.exists():
        raise FileNotFoundError(
            f"ARC root {root_path} not found. Run scripts/sync_arc_agi_2.sh first."
        )

    tasks: list[ARCTask] = []
    per_file: dict[str, str] = {}
    counts = {"training": 0, "evaluation": 0}

    for split in ("training", "evaluation"):
        split_dir = _split_dir(root_path, split)
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected split dir missing: {split_dir}")
        files = sorted(split_dir.glob("*.json"))
        for fp in files:
            task = load_task(fp, split)
            tasks.append(task)
            rel = str(fp.relative_to(root_path))
            per_file[rel] = file_sha256(fp)
            counts[split] += 1
        logger.info("Ingested %d %s tasks from %s", counts[split], split, split_dir)

    manifest = {
        "dataset_name": DATASET_NAME,
        "source_repo_path": str(root_path),
        "num_training_tasks": counts["training"],
        "num_evaluation_tasks": counts["evaluation"],
        "num_total_tasks": counts["training"] + counts["evaluation"],
        "per_file_sha256": per_file,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "code_git_commit": git_commit(REPO_ROOT),
    }
    manifest["dataset_hash"] = dict_sha256(per_file)
    return tasks, manifest


def write_manifest(tasks: list[ARCTask], manifest: dict, manifest_path: str | Path = MANIFEST_PATH) -> Path:
    out = write_json(manifest_path, manifest)
    logger.info(
        "Wrote manifest with %d tasks (dataset_hash=%s) -> %s",
        len(tasks),
        manifest["dataset_hash"][:12],
        out,
    )
    return out


def main(argv: list[str]) -> int:
    root = argv[1] if len(argv) > 1 else "data/raw/ARC-AGI-2"
    tasks, manifest = ingest_arc(root)
    write_manifest(tasks, manifest)
    print(
        f"OK: {manifest['num_training_tasks']} training, "
        f"{manifest['num_evaluation_tasks']} evaluation tasks validated. "
        f"dataset_hash={manifest['dataset_hash'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
