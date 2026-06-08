"""Model utilities: deterministic seeding and a config-driven model factory."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from packages.model.baseline import BaselineARCModel
from packages.model.trm_arc import TRMARCModel


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Build a model from the ``model`` section of a config dict.

    ``arch`` selects ``trm`` (default) or ``baseline``.
    """
    cfg = dict(model_cfg)
    arch = cfg.pop("arch", "trm")
    cfg.pop("name", None)
    if arch == "baseline":
        return BaselineARCModel(**cfg)
    if arch == "trm":
        cfg.pop("num_layers", None)  # not used by recursive model
        return TRMARCModel(**cfg)
    raise ValueError(f"Unknown model arch: {arch!r}")
