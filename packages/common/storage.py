"""Small filesystem helpers for artifacts, reports, and JSON IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from packages.common.config import REPO_ROOT


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
        f.write("\n")
    return p


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
