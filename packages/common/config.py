"""Centralized configuration.

Two layers:
- ``Settings``: process/environment-level settings (DB URL, paths, device) loaded
  from environment variables / ``.env`` via pydantic-settings.
- ``load_yaml_config``: experiment configs (model/data/training/eval) loaded from
  the YAML files under ``configs/``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-level settings.

    ``DATABASE_URL`` defaults to a local SQLite file so tests and local runs work
    without a running Postgres. Docker Compose overrides it to point at Postgres.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = f"sqlite:///{REPO_ROOT / 'arc_engine.db'}"
    redis_url: str = "redis://localhost:6379/0"
    arc_data_root: str = "data/raw/ARC-AGI-2"
    model_registry_path: str = "artifacts/model_registry.json"
    checkpoint_dir: str = "checkpoints"
    max_recursion_steps: int = 64
    default_model_id: str = "trm_arc_v1"
    arc_device: str = "auto"
    api_base_url: str = "http://localhost:8080"

    @property
    def data_root_path(self) -> Path:
        p = Path(self.arc_data_root)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def registry_path(self) -> Path:
        p = Path(self.model_registry_path)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def checkpoint_path(self) -> Path:
        p = Path(self.checkpoint_dir)
        return p if p.is_absolute() else REPO_ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML config into a plain dict."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} did not parse to a mapping")
    return data


def resolve_device(preference: str | None = None) -> str:
    """Resolve the torch device string given a preference (auto/cpu/cuda/mps)."""
    import torch

    pref = (preference or get_settings().arc_device or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and os.environ.get("ARC_DISABLE_MPS") != "1":
        return "mps"
    return "cpu"
