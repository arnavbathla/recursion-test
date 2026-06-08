"""Shared pytest fixtures.

Most tests use REAL ARC-AGI-2 data (after sync). If the data is not present, those
tests are skipped with a clear message (rather than silently passing on fake data).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARC_ROOT = REPO_ROOT / "data" / "raw" / "ARC-AGI-2"

# Force CPU + an isolated SQLite DB for the whole test session BEFORE importing
# any project module that reads settings.
os.environ.setdefault("ARC_DEVICE", "cpu")
os.environ["ARC_DISABLE_MPS"] = "1"
_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="arc_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"


def arc_available() -> bool:
    return (ARC_ROOT / "data" / "training").exists()


requires_arc = pytest.mark.skipif(
    not arc_available(),
    reason="ARC-AGI-2 data not synced. Run scripts/sync_arc_agi_2.sh",
)


@pytest.fixture(scope="session")
def arc_tasks():
    if not arc_available():
        pytest.skip("ARC-AGI-2 data not synced")
    from packages.data.ingest_arc import ingest_arc

    tasks, manifest = ingest_arc(str(ARC_ROOT))
    return tasks, manifest


@pytest.fixture(scope="session")
def training_tasks(arc_tasks):
    tasks, _ = arc_tasks
    return [t for t in tasks if t.split == "training"]


@pytest.fixture(scope="session")
def small_dataset(training_tasks):
    from packages.data.dataset import ARCDataset

    return ARCDataset(training_tasks[:8], "train")


def pytest_sessionfinish(session, exitstatus):
    try:
        os.close(_DB_FD)
    except OSError:
        pass
    try:
        os.unlink(_DB_PATH)
    except OSError:
        pass
