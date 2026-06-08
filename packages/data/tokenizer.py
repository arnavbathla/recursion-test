"""Tokenizer / packer: turn an ARCTask into a flat token sequence for the model.

Vocabulary (VOCAB_SIZE = 19):
    0-9  : grid colors
    10   : PAD
    11   : BOS
    12   : EOS
    13   : TRAIN_INPUT marker
    14   : TRAIN_OUTPUT marker
    15   : TEST_INPUT marker
    16   : ANSWER marker
    17   : ROW_SEP
    18   : GRID_SEP

Packed layout:
    BOS
    [ for each train pair:
        TRAIN_INPUT  + <flattened input grid>  + GRID_SEP
        TRAIN_OUTPUT + <flattened output grid> + GRID_SEP ]
    TEST_INPUT + <flattened test input grid> + GRID_SEP
    ANSWER + 900 PAD slots         (the answer canvas; row-major 30x30)
    EOS

The 900 answer slots form a 30x30 row-major canvas. The target output grid is
written into the top-left corner of that canvas; ``answer_positions`` records the
sequence index of each of the 900 slots.

No ``task_id`` (or any stable identifier) is ever encoded as a token or feature.
Sequences that exceed ``MAX_SEQ_LEN`` raise ``SequenceTooLongError`` (no silent
truncation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from packages.data.schema import ARCTask, Grid

# --- color tokens ---
PAD = 10
BOS = 11
EOS = 12
TRAIN_INPUT = 13
TRAIN_OUTPUT = 14
TEST_INPUT = 15
ANSWER = 16
ROW_SEP = 17
GRID_SEP = 18

MAX_GRID = 30
MAX_CELLS = MAX_GRID * MAX_GRID  # 900
VOCAB_SIZE = 19
MAX_SEQ_LEN = 4096
NO_COORD = MAX_GRID + 1  # 31: sentinel row/col for non-cell tokens

# segment ids (model uses num_segments=8)
SEG_CONTROL = 0
SEG_TRAIN_INPUT = 1
SEG_TRAIN_OUTPUT = 2
SEG_TEST_INPUT = 3
SEG_ANSWER = 4
SEG_GRID_SEP = 5
SEG_ROW_SEP = 6


class SequenceTooLongError(ValueError):
    """Raised when a packed task exceeds MAX_SEQ_LEN."""


@dataclass
class PackedTask:
    tokens: torch.Tensor          # [S] long
    rows: torch.Tensor            # [S] long
    cols: torch.Tensor            # [S] long
    segments: torch.Tensor        # [S] long
    answer_positions: torch.Tensor  # [900] long, indices into tokens

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[0])


def flatten_grid(grid: Grid) -> list[tuple[int, int, int]]:
    """Flatten a grid to a list of (token, row, col), row-major, with ROW_SEP
    between consecutive rows (not after the final row)."""
    out: list[tuple[int, int, int]] = []
    height = len(grid)
    for r in range(height):
        row = grid[r]
        for c, val in enumerate(row):
            out.append((int(val), r, c))
        if r != height - 1:
            out.append((ROW_SEP, NO_COORD, NO_COORD))
    return out


class _Builder:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.segments: list[int] = []

    def add(self, token: int, row: int, col: int, seg: int) -> int:
        idx = len(self.tokens)
        self.tokens.append(token)
        self.rows.append(row)
        self.cols.append(col)
        self.segments.append(seg)
        return idx

    def add_grid(self, grid: Grid, seg: int) -> None:
        for tok, r, c in flatten_grid(grid):
            s = SEG_ROW_SEP if tok == ROW_SEP else seg
            self.add(tok, r, c, s)


def pack_task(task: ARCTask, test_index: int = 0) -> PackedTask:
    """Pack a single ARC task (using one test pair) into model token tensors."""
    if test_index < 0 or test_index >= len(task.test):
        raise IndexError(f"test_index {test_index} out of range for task {task.task_id}")

    b = _Builder()
    b.add(BOS, NO_COORD, NO_COORD, SEG_CONTROL)

    for pair in task.train:
        b.add(TRAIN_INPUT, NO_COORD, NO_COORD, SEG_TRAIN_INPUT)
        b.add_grid(pair.input, SEG_TRAIN_INPUT)
        b.add(GRID_SEP, NO_COORD, NO_COORD, SEG_GRID_SEP)

        assert pair.output is not None  # guaranteed by schema for train pairs
        b.add(TRAIN_OUTPUT, NO_COORD, NO_COORD, SEG_TRAIN_OUTPUT)
        b.add_grid(pair.output, SEG_TRAIN_OUTPUT)
        b.add(GRID_SEP, NO_COORD, NO_COORD, SEG_GRID_SEP)

    test_input = task.test[test_index].input
    b.add(TEST_INPUT, NO_COORD, NO_COORD, SEG_TEST_INPUT)
    b.add_grid(test_input, SEG_TEST_INPUT)
    b.add(GRID_SEP, NO_COORD, NO_COORD, SEG_GRID_SEP)

    b.add(ANSWER, NO_COORD, NO_COORD, SEG_ANSWER)
    answer_positions: list[int] = []
    for i in range(MAX_CELLS):
        r, c = divmod(i, MAX_GRID)
        idx = b.add(PAD, r, c, SEG_ANSWER)
        answer_positions.append(idx)

    b.add(EOS, NO_COORD, NO_COORD, SEG_CONTROL)

    seq_len = len(b.tokens)
    if seq_len > MAX_SEQ_LEN:
        raise SequenceTooLongError(
            f"Packed task {task.task_id} has length {seq_len} > MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            f"Refusing to truncate."
        )

    return PackedTask(
        tokens=torch.tensor(b.tokens, dtype=torch.long),
        rows=torch.tensor(b.rows, dtype=torch.long),
        cols=torch.tensor(b.cols, dtype=torch.long),
        segments=torch.tensor(b.segments, dtype=torch.long),
        answer_positions=torch.tensor(answer_positions, dtype=torch.long),
    )


def pad_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of per-example tensor dicts into a batched dict.

    Pads ``tokens``/``rows``/``cols``/``segments`` to the max length in the batch
    and builds ``attention_mask`` (True == padding, matching nn.MultiheadAttention's
    ``key_padding_mask`` convention). ``answer_positions`` are fixed-length (900)
    and stacked directly.
    """
    if not batch:
        raise ValueError("pad_batch received an empty batch")

    max_len = max(int(item["tokens"].shape[0]) for item in batch)
    bsz = len(batch)

    tokens = torch.full((bsz, max_len), PAD, dtype=torch.long)
    rows = torch.full((bsz, max_len), NO_COORD, dtype=torch.long)
    cols = torch.full((bsz, max_len), NO_COORD, dtype=torch.long)
    segments = torch.full((bsz, max_len), SEG_CONTROL, dtype=torch.long)
    attention_mask = torch.ones((bsz, max_len), dtype=torch.bool)  # True = pad

    for i, item in enumerate(batch):
        s = int(item["tokens"].shape[0])
        tokens[i, :s] = item["tokens"]
        rows[i, :s] = item["rows"]
        cols[i, :s] = item["cols"]
        segments[i, :s] = item["segments"]
        attention_mask[i, :s] = False

    out: dict[str, torch.Tensor] = {
        "tokens": tokens,
        "rows": rows,
        "cols": cols,
        "segments": segments,
        "attention_mask": attention_mask,
        "answer_positions": torch.stack([item["answer_positions"] for item in batch]),
    }
    return out
