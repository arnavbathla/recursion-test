"""Objective ARC grid verifier.

Exact match is the primary ARC metric: prediction must equal the target at every
cell with identical shape. Pixel accuracy is a softer, shape-gated signal.
"""

from __future__ import annotations

Grid = list[list[int]]

MAX_GRID = 30


def validate_grid(grid: object) -> bool:
    """Return True iff ``grid`` is a well-formed ARC grid (rectangular, 1-30, ints 0-9)."""
    if not isinstance(grid, list) or len(grid) == 0 or len(grid) > MAX_GRID:
        return False
    width: int | None = None
    for row in grid:
        if not isinstance(row, list) or len(row) == 0 or len(row) > MAX_GRID:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for val in row:
            if isinstance(val, bool) or not isinstance(val, int):
                return False
            if val < 0 or val > 9:
                return False
    return True


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def shape_accuracy(pred: Grid, target: Grid) -> bool:
    """True iff predicted shape equals target shape."""
    return _shape(pred) == _shape(target)


def exact_match(pred: Grid, target: Grid) -> bool:
    """True iff prediction equals target exactly (shape + all cells)."""
    if not shape_accuracy(pred, target):
        return False
    h, w = _shape(target)
    for r in range(h):
        prow = pred[r]
        trow = target[r]
        for c in range(w):
            if prow[c] != trow[c]:
                return False
    return True


def pixel_accuracy(pred: Grid, target: Grid) -> float:
    """Fraction of matching cells. Returns 0.0 on any shape mismatch."""
    if not shape_accuracy(pred, target):
        return 0.0
    h, w = _shape(target)
    total = h * w
    if total == 0:
        return 0.0
    correct = 0
    for r in range(h):
        prow = pred[r]
        trow = target[r]
        for c in range(w):
            if prow[c] == trow[c]:
                correct += 1
    return correct / total
