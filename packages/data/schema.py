"""ARC task schema with strict validation.

Grids are validated, not coerced: a malformed grid raises rather than being
silently repaired. Values must be ints in [0, 9]; grids must be rectangular with
height/width in [1, 30].
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

MAX_GRID = 30
MIN_GRID = 1

Grid = list[list[int]]
Split = Literal["training", "evaluation"]


def validate_grid_values(grid: object) -> Grid:
    """Validate raw grid structure/values. Raises ValueError on any violation."""
    if not isinstance(grid, list) or len(grid) == 0:
        raise ValueError("grid must be a non-empty list of rows")
    height = len(grid)
    if height < MIN_GRID or height > MAX_GRID:
        raise ValueError(f"grid height {height} out of range [1, {MAX_GRID}]")
    width: int | None = None
    for r, row in enumerate(grid):
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError(f"row {r} must be a non-empty list")
        if width is None:
            width = len(row)
            if width < MIN_GRID or width > MAX_GRID:
                raise ValueError(f"grid width {width} out of range [1, {MAX_GRID}]")
        elif len(row) != width:
            raise ValueError(f"grid is not rectangular: row {r} has width {len(row)} != {width}")
        for c, val in enumerate(row):
            # bool is a subclass of int; reject it explicitly to avoid silent coercion.
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"cell ({r},{c})={val!r} is not an int")
            if val < 0 or val > 9:
                raise ValueError(f"cell ({r},{c})={val} out of range [0, 9]")
    return grid  # type: ignore[return-value]


class ARCPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Grid
    output: Grid | None = None

    @field_validator("input")
    @classmethod
    def _check_input(cls, v: Grid) -> Grid:
        return validate_grid_values(v)

    @field_validator("output")
    @classmethod
    def _check_output(cls, v: Grid | None) -> Grid | None:
        if v is None:
            return None
        return validate_grid_values(v)


class ARCTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    split: Split
    train: list[ARCPair]
    test: list[ARCPair]

    @model_validator(mode="after")
    def _check_pairs(self) -> ARCTask:
        if len(self.train) == 0:
            raise ValueError(f"task {self.task_id} has no train pairs")
        if len(self.test) == 0:
            raise ValueError(f"task {self.task_id} has no test pairs")
        for i, pair in enumerate(self.train):
            if pair.output is None:
                raise ValueError(f"task {self.task_id} train pair {i} missing output")
        return self

    @property
    def num_train_pairs(self) -> int:
        return len(self.train)

    @property
    def num_test_pairs(self) -> int:
        return len(self.test)

    def to_raw(self) -> dict:
        """Serialize back to the ARC on-disk shape (train/test pairs only)."""
        return {
            "train": [
                {"input": p.input, "output": p.output} for p in self.train
            ],
            "test": [
                ({"input": p.input, "output": p.output} if p.output is not None else {"input": p.input})
                for p in self.test
            ],
        }
