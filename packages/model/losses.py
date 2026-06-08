"""Loss function for the ARC heads.

Total loss = height CE + width CE + cell-color CE (over valid output cells only).
- Cell loss ignores slots with target value 10 (the CELL_IGNORE sentinel).
- Height/width CE uses torch's default ignore_index=-100 so examples without a
  target (no output grid) are skipped for shape loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from packages.model.trm_arc import TRMARCOutput

CELL_IGNORE = 10
SHAPE_IGNORE = -100


def arc_loss(outputs: TRMARCOutput, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    target_height = batch["target_height"]
    target_width = batch["target_width"]
    target_cells = batch["target_cells"]

    height_loss = F.cross_entropy(
        outputs.height_logits, target_height, ignore_index=SHAPE_IGNORE
    )
    width_loss = F.cross_entropy(
        outputs.width_logits, target_width, ignore_index=SHAPE_IGNORE
    )

    cell_logits = outputs.cell_logits.reshape(-1, 10)
    cell_targets = target_cells.reshape(-1)
    valid = cell_targets != CELL_IGNORE
    if valid.any():
        cell_loss = F.cross_entropy(cell_logits[valid], cell_targets[valid])
    else:
        cell_loss = cell_logits.sum() * 0.0

    # Guard: if an entire batch lacks shape targets, CE returns NaN. Replace with 0.
    if torch.isnan(height_loss):
        height_loss = cell_loss.new_zeros(())
    if torch.isnan(width_loss):
        width_loss = cell_loss.new_zeros(())

    loss = height_loss + width_loss + cell_loss

    metrics = {
        "loss": float(loss.detach()),
        "height_loss": float(height_loss.detach()),
        "width_loss": float(width_loss.detach()),
        "cell_loss": float(cell_loss.detach()),
        "num_valid_cells": int(valid.sum().detach()),
    }
    return loss, metrics
