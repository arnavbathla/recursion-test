"""Decode model logits into concrete ARC grids.

The answer canvas is a 30x30 row-major grid. The predicted height/width select the
top-left sub-grid; cell colors are argmaxed over the 10 color classes.
"""

from __future__ import annotations

import torch

from packages.eval.verifier import validate_grid
from packages.model.trm_arc import MAX_GRID, TRMARCOutput

Grid = list[list[int]]


def decode_outputs(outputs: TRMARCOutput, batch_index: int = 0) -> Grid:
    """Decode a single example's logits into a validated grid."""
    height = int(torch.argmax(outputs.height_logits[batch_index]).item()) + 1
    width = int(torch.argmax(outputs.width_logits[batch_index]).item()) + 1
    height = max(1, min(MAX_GRID, height))
    width = max(1, min(MAX_GRID, width))

    cell_pred = torch.argmax(outputs.cell_logits[batch_index], dim=-1)  # [900]
    grid: Grid = []
    for r in range(height):
        row: list[int] = []
        for c in range(width):
            row.append(int(cell_pred[r * MAX_GRID + c].item()))
        grid.append(row)

    if not validate_grid(grid):
        raise ValueError(f"Decoded grid failed validation (h={height}, w={width})")
    return grid


def decode_trace(trace: list[TRMARCOutput], batch_index: int = 0) -> list[dict]:
    """Decode intermediate trace outputs into step-tagged grids."""
    trace_step_labels = [1, 2, 4, 8, 16, 32, 64]
    items: list[dict] = []
    for i, out in enumerate(trace):
        grid = decode_outputs(out, batch_index=batch_index)
        step = trace_step_labels[i] if i < len(trace_step_labels) else i + 1
        items.append(
            {
                "step": step,
                "height": len(grid),
                "width": len(grid[0]),
                "grid": grid,
            }
        )
    return items
