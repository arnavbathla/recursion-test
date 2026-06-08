"""PyTorch Dataset + collate for packed ARC tasks.

Each example corresponds to one (task, test_index) pair. Targets:
- ``target_height``/``target_width``: class index 0-29 meaning height/width 1-30
  (or ``IGNORE_INDEX`` when the test pair has no output).
- ``target_cells``: [900] long over the 30x30 row-major answer canvas; real output
  cells are 0-9, all other slots use ``CELL_IGNORE`` (10).

Tasks whose packed length exceeds MAX_SEQ_LEN are explicitly excluded at
construction time (with a logged count); they are never silently truncated.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from packages.common.logging import get_logger
from packages.data.schema import ARCTask
from packages.data.tokenizer import (
    MAX_CELLS,
    MAX_GRID,
    SequenceTooLongError,
    pack_task,
    pad_batch,
)

logger = get_logger(__name__)

CELL_IGNORE = 10       # ignore index for cell-color loss (matches PAD color slot)
IGNORE_INDEX = -100    # ignore index for height/width CE (torch default)


def build_target_cells(output: list[list[int]] | None) -> tuple[torch.Tensor, int, int]:
    """Return (target_cells[900], target_height_class, target_width_class).

    Output grid placed at the top-left of a 30x30 row-major canvas. Slots outside
    the output grid use ``CELL_IGNORE``. If ``output`` is None, all cells ignore and
    height/width classes are ``IGNORE_INDEX``.
    """
    target = torch.full((MAX_CELLS,), CELL_IGNORE, dtype=torch.long)
    if output is None:
        return target, IGNORE_INDEX, IGNORE_INDEX

    h = len(output)
    w = len(output[0])
    for r in range(h):
        for c in range(w):
            target[r * MAX_GRID + c] = int(output[r][c])
    return target, h - 1, w - 1


class ARCDataset(Dataset):
    """Dataset of packed ARC examples built from a list of ARCTask."""

    def __init__(self, tasks: list[ARCTask], split: str) -> None:
        self.split = split
        self.examples: list[dict] = []
        skipped = 0
        for task in tasks:
            for test_index in range(task.num_test_pairs):
                try:
                    packed = pack_task(task, test_index=test_index)
                except SequenceTooLongError:
                    skipped += 1
                    continue
                output = task.test[test_index].output
                target_cells, th, tw = build_target_cells(output)
                self.examples.append(
                    {
                        "tokens": packed.tokens,
                        "rows": packed.rows,
                        "cols": packed.cols,
                        "segments": packed.segments,
                        "answer_positions": packed.answer_positions,
                        "target_cells": target_cells,
                        "target_height": torch.tensor(th, dtype=torch.long),
                        "target_width": torch.tensor(tw, dtype=torch.long),
                        "has_target": output is not None,
                        "task_id": task.task_id,
                        "split": split,
                    }
                )
        if skipped:
            logger.warning(
                "ARCDataset(%s): skipped %d example(s) exceeding MAX_SEQ_LEN", split, skipped
            )
        logger.info("ARCDataset(%s): %d examples", split, len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate(batch: list[dict]) -> dict:
    """Collate function: pad sequences and stack targets/metadata."""
    padded = pad_batch(batch)
    padded["target_cells"] = torch.stack([b["target_cells"] for b in batch])
    padded["target_height"] = torch.stack([b["target_height"] for b in batch])
    padded["target_width"] = torch.stack([b["target_width"] for b in batch])
    padded["has_target"] = torch.tensor([b["has_target"] for b in batch], dtype=torch.bool)
    padded["task_id"] = [b["task_id"] for b in batch]
    padded["split"] = [b["split"] for b in batch]
    return padded
