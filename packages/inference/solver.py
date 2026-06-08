"""High-level solve: run a model on an ARCTask and produce a structured result.

This is pure inference logic (no I/O, no DB). The runtime layer wraps it with
checkpoint loading and caching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from packages.data.dataset import collate
from packages.data.schema import ARCTask
from packages.data.tokenizer import pack_task
from packages.eval.verifier import exact_match, pixel_accuracy, shape_accuracy
from packages.inference.decode import decode_outputs, decode_trace

Grid = list[list[int]]


@dataclass
class SolveResult:
    prediction: Grid
    trace: list[dict] = field(default_factory=list)
    exact_match: bool | None = None
    pixel_accuracy: float | None = None
    shape_accuracy: bool | None = None
    latency_ms: float = 0.0
    recursion_steps: int = 0


@torch.no_grad()
def solve_task(
    model: nn.Module,
    task: ARCTask,
    *,
    recursion_steps: int = 16,
    return_trace: bool = False,
    test_index: int = 0,
    device: str = "cpu",
) -> SolveResult:
    model.eval()
    packed = pack_task(task, test_index=test_index)
    batch = collate(
        [
            {
                "tokens": packed.tokens,
                "rows": packed.rows,
                "cols": packed.cols,
                "segments": packed.segments,
                "answer_positions": packed.answer_positions,
                "target_cells": torch.full((900,), 10, dtype=torch.long),
                "target_height": torch.tensor(-100),
                "target_width": torch.tensor(-100),
                "has_target": False,
                "task_id": task.task_id,
                "split": task.split,
            }
        ]
    )
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    t0 = time.perf_counter()
    forward_out = model(
        tokens=batch["tokens"],
        rows=batch["rows"],
        cols=batch["cols"],
        segments=batch["segments"],
        answer_positions=batch["answer_positions"],
        attention_mask=batch["attention_mask"],
        recursion_steps=recursion_steps,
        return_trace=return_trace,
    )
    if return_trace:
        out, trace_outputs = forward_out
    else:
        out, trace_outputs = forward_out, []

    prediction = decode_outputs(out, batch_index=0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    trace = decode_trace(trace_outputs, batch_index=0) if return_trace else []

    result = SolveResult(
        prediction=prediction,
        trace=trace,
        latency_ms=round(latency_ms, 3),
        recursion_steps=recursion_steps,
    )

    target = task.test[test_index].output
    if target is not None:
        result.exact_match = exact_match(prediction, target)
        result.pixel_accuracy = round(pixel_accuracy(prediction, target), 6)
        result.shape_accuracy = shape_accuracy(prediction, target)

    return result


def result_to_dict(result: SolveResult) -> dict[str, Any]:
    return {
        "prediction": result.prediction,
        "trace": result.trace,
        "exact_match": result.exact_match,
        "pixel_accuracy": result.pixel_accuracy,
        "shape_accuracy": result.shape_accuracy,
        "latency_ms": result.latency_ms,
        "recursion_steps": result.recursion_steps,
    }
