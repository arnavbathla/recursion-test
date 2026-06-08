"""Deterministic hashing utilities for data manifests, configs, and checkpoints."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file and return its hex sha256."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dict_sha256(obj: Any) -> str:
    """Hash an arbitrary JSON-serializable object deterministically."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit(repo_root: str | Path | None = None) -> str | None:
    """Best-effort current git commit hash; ``None`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None
