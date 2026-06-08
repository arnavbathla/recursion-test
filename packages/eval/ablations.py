"""Ablation suite comparing recursion, depth-1, carry-disabled, baseline, and a
shuffled-example-order control.

These ablations isolate the effect of recursive refinement and verify the model is
not exploiting spurious cues (e.g. example ordering).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch

from packages.common.config import REPO_ROOT, load_yaml_config, resolve_device
from packages.common.logging import get_logger
from packages.common.storage import read_json, write_json
from packages.data.dataset import ARCDataset, collate
from packages.data.ingest_arc import MANIFEST_PATH, ingest_arc
from packages.data.splits import select_split
from packages.eval.metrics import aggregate
from packages.eval.verifier import exact_match, pixel_accuracy, shape_accuracy
from packages.inference.decode import decode_outputs
from packages.training.checkpoint import build_model_from_checkpoint, load_checkpoint
from packages.training.trainer import reconstruct_target_grid

logger = get_logger(__name__)


def _dataset_for(config: dict[str, Any], split: str) -> ARCDataset:
    data_cfg = config.get("data", {})
    tasks, _ = ingest_arc(data_cfg.get("root", "data/raw/ARC-AGI-2"))
    selected = select_split(
        tasks, split,
        holdout_ratio=data_cfg.get("train_holdout_ratio", 0.15),
        seed=data_cfg.get("seed", 42),
        allow_official_eval=(split == "evaluation"),
        for_training=False,
    )
    return ARCDataset(selected, split)


def _shuffle_example_tasks(tasks):
    """Return new tasks with train-example order shuffled (control ablation)."""
    out = []
    rng = random.Random(123)
    for t in tasks:
        t2 = t.model_copy(deep=True)
        rng.shuffle(t2.train)
        out.append(t2)
    return out


@torch.no_grad()
def _score(model, examples, depth: int, device: str) -> dict:
    model.eval()
    exacts, pixels, shapes, latencies = [], [], [], []
    for ex in examples:
        if not ex["has_target"]:
            continue
        batch = collate([ex])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        t0 = time.perf_counter()
        out = model(
            tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
            segments=batch["segments"], answer_positions=batch["answer_positions"],
            attention_mask=batch["attention_mask"], recursion_steps=depth,
        )
        pred = decode_outputs(out, 0)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        target = reconstruct_target_grid(ex["target_cells"], int(ex["target_height"]), int(ex["target_width"]))
        exacts.append(exact_match(pred, target))
        pixels.append(pixel_accuracy(pred, target))
        shapes.append(shape_accuracy(pred, target))
    return aggregate(exacts, pixels, shapes, latencies).to_dict()


def run_ablations(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    split: str = "train_holdout",
    full_depth: int = 16,
    limit: int | None = None,
    baseline_checkpoint: str | Path | None = None,
    baseline_config: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device()
    model = build_model_from_checkpoint(load_checkpoint(checkpoint_path)).to(device)
    dataset = _dataset_for(config, split)
    examples = dataset.examples if limit is None else dataset.examples[:limit]

    ablations: list[dict[str, Any]] = []

    ablations.append({
        "model": "recursive", "recursion_depth": full_depth,
        "notes": "normal recursive model at full depth",
        **_score(model, examples, full_depth, device),
    })
    ablations.append({
        "model": "recursive", "recursion_depth": 1,
        "notes": "recursive model forced to a single recursive step",
        **_score(model, examples, 1, device),
    })

    # Shuffled-example-order control (same model, shuffled train demos).
    data_cfg = config.get("data", {})
    tasks, _ = ingest_arc(data_cfg.get("root", "data/raw/ARC-AGI-2"))
    sel = select_split(tasks, split, holdout_ratio=data_cfg.get("train_holdout_ratio", 0.15),
                       seed=data_cfg.get("seed", 42), allow_official_eval=(split == "evaluation"),
                       for_training=False)
    shuffled_ds = ARCDataset(_shuffle_example_tasks(sel), split)
    shuffled_examples = shuffled_ds.examples if limit is None else shuffled_ds.examples[:limit]
    ablations.append({
        "model": "recursive", "recursion_depth": full_depth,
        "notes": "shuffled train-example order (robustness control)",
        **_score(model, shuffled_examples, full_depth, device),
    })

    if baseline_checkpoint and baseline_config:
        bmodel = build_model_from_checkpoint(load_checkpoint(baseline_checkpoint)).to(device)
        ablations.append({
            "model": "baseline", "recursion_depth": 1,
            "notes": "non-recursive baseline, single forward pass",
            **_score(bmodel, examples, 1, device),
        })

    report = {
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "device": device,
        "dataset_hash": (read_json(REPO_ROOT / MANIFEST_PATH).get("dataset_hash")
                         if (REPO_ROOT / MANIFEST_PATH).exists() else None),
        "ablations": ablations,
    }
    if out_path:
        write_json(out_path, report)
        logger.info("Wrote ablation report -> %s", out_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation suite")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train_holdout")
    parser.add_argument("--full-depth", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    baseline_config = load_yaml_config(args.baseline_config) if args.baseline_config else None
    run_ablations(
        args.checkpoint, config, split=args.split, full_depth=args.full_depth, limit=args.limit,
        baseline_checkpoint=args.baseline_checkpoint, baseline_config=baseline_config,
        out_path=args.out,
    )
    print("OK: ablations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
