"""Inference runtime: load checkpoints into memory and serve solves.

A ``ModelRuntime`` loads a checkpoint once and caches the model (keyed by model_id),
so the API does not reload weights per request. Resolves the device (CUDA > MPS >
CPU). Supports a CPU mode for tests and a CUDA mode for production.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from packages.common.config import REPO_ROOT, get_settings, resolve_device
from packages.common.hashing import file_sha256
from packages.common.logging import get_logger
from packages.common.registry import ModelRecord, get_record
from packages.data.schema import ARCTask
from packages.inference.solver import SolveResult, solve_task
from packages.model.trm_arc import count_parameters
from packages.training.checkpoint import build_model_from_checkpoint, load_checkpoint

logger = get_logger(__name__)


class CheckpointUnavailableError(RuntimeError):
    """Raised when a model's checkpoint cannot be found/loaded."""


@dataclass
class LoadedModel:
    model_id: str
    model: nn.Module
    checkpoint_path: str
    checkpoint_hash: str
    dataset_hash: str | None
    param_count: int


class ModelRuntime:
    """Loads and caches models for inference. Thread-safe for the inline API path."""

    def __init__(self, device: str | None = None, dev_reload: bool = False) -> None:
        self.device = device or resolve_device()
        self.dev_reload = dev_reload
        self._cache: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        logger.info("ModelRuntime initialized on device=%s", self.device)

    def _resolve_checkpoint(self, model_id: str) -> tuple[Path, ModelRecord | None]:
        record = get_record(model_id)
        if record is not None:
            ckpt = Path(record.checkpoint_path)
            if not ckpt.is_absolute():
                ckpt = REPO_ROOT / ckpt
            if ckpt.exists():
                return ckpt, record
        # Fall back to <checkpoint_dir>/<model_id>.pt
        fallback = get_settings().checkpoint_path / f"{model_id}.pt"
        if fallback.exists():
            return fallback, record
        raise CheckpointUnavailableError(
            f"No checkpoint found for model_id={model_id!r}. Train it first "
            f"(e.g. bash scripts/smoke_train.sh) or register it in the model registry."
        )

    def load(self, model_id: str) -> LoadedModel:
        if not self.dev_reload:
            cached = self._cache.get(model_id)
            if cached is not None:
                return cached
        with self._lock:
            cached = self._cache.get(model_id)
            if cached is not None and not self.dev_reload:
                return cached
            ckpt_path, _ = self._resolve_checkpoint(model_id)
            payload = load_checkpoint(ckpt_path, map_location=self.device)
            model = build_model_from_checkpoint(payload).to(self.device)
            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                checkpoint_path=str(ckpt_path),
                checkpoint_hash=file_sha256(ckpt_path),
                dataset_hash=payload.get("dataset_hash"),
                param_count=count_parameters(model),
            )
            self._cache[model_id] = loaded
            logger.info(
                "Loaded model %s (%d params) from %s on %s",
                model_id, loaded.param_count, ckpt_path, self.device,
            )
            return loaded

    def solve_task(
        self,
        task: ARCTask,
        *,
        model_id: str,
        recursion_steps: int = 16,
        return_trace: bool = False,
        test_index: int = 0,
    ) -> tuple[SolveResult, LoadedModel]:
        loaded = self.load(model_id)
        result = solve_task(
            loaded.model,
            task,
            recursion_steps=recursion_steps,
            return_trace=return_trace,
            test_index=test_index,
            device=self.device,
        )
        return result, loaded

    def info(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "loaded_models": list(self._cache.keys()),
        }


_GLOBAL_RUNTIME: ModelRuntime | None = None


def get_runtime() -> ModelRuntime:
    global _GLOBAL_RUNTIME
    if _GLOBAL_RUNTIME is None:
        _GLOBAL_RUNTIME = ModelRuntime()
    return _GLOBAL_RUNTIME
