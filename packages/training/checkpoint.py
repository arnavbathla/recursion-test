"""Checkpoint save/load with embedded config + provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from packages.common.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict[str, Any],
    *,
    epoch: int | None = None,
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
    dataset_hash: str | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "model_config": getattr(model, "config", {}),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "dataset_hash": dataset_hash,
        "arch": config.get("model", {}).get("arch", "trm"),
    }
    torch.save(payload, p)
    logger.info("Saved checkpoint -> %s", p)
    return p


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return torch.load(p, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> nn.Module:
    """Reconstruct the model from a checkpoint payload and load weights."""
    from packages.model.utils import build_model

    model_cfg = dict(ckpt["config"].get("model", {}))
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
