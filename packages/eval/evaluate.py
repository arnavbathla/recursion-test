"""Recursion-depth evaluation harness.

For each requested depth, runs the model over every example in a split (batch of 1
to measure per-example latency), decodes predictions, and computes exact-match /
pixel-accuracy / shape-accuracy / latency percentiles. Writes a JSON report.

CLI:
    python -m packages.eval.evaluate \
        --checkpoint checkpoints/trm_arc_v1_best.pt \
        --config configs/trm_arc_v1.yaml \
        --split train_holdout \
        --depths 1 2 4 8 16 32 64 \
        --out reports/trm_arc_v1_eval.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from packages.common.config import REPO_ROOT, resolve_device
from packages.common.hashing import dict_sha256, file_sha256
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


def _dataset_hash() -> str | None:
    p = REPO_ROOT / MANIFEST_PATH
    return read_json(p).get("dataset_hash") if p.exists() else None


def _build_split_dataset(config: dict[str, Any], split: str) -> ARCDataset:
    data_cfg = config.get("data", {})
    tasks, _ = ingest_arc(data_cfg.get("root", "data/raw/ARC-AGI-2"))
    # Final evaluation on the official evaluation split is permitted here (not training).
    allow_eval = split == "evaluation"
    selected = select_split(
        tasks,
        split,
        holdout_ratio=data_cfg.get("train_holdout_ratio", 0.15),
        seed=data_cfg.get("seed", 42),
        allow_official_eval=allow_eval,
        for_training=False,
    )
    return ARCDataset(selected, split)


@torch.no_grad()
def evaluate_depths(
    model: torch.nn.Module,
    dataset: ARCDataset,
    depths: list[int],
    device: str,
    limit: int | None = None,
) -> dict[str, dict]:
    model.eval()
    model.to(device)
    examples = dataset.examples if limit is None else dataset.examples[:limit]

    results: dict[str, dict] = {}
    for depth in depths:
        exacts: list[bool] = []
        pixels: list[float] = []
        shapes: list[bool] = []
        latencies: list[float] = []
        for ex in examples:
            if not ex["has_target"]:
                continue
            batch = collate([ex])
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            t0 = time.perf_counter()
            out = model(
                tokens=batch["tokens"],
                rows=batch["rows"],
                cols=batch["cols"],
                segments=batch["segments"],
                answer_positions=batch["answer_positions"],
                attention_mask=batch["attention_mask"],
                recursion_steps=depth,
            )
            pred = decode_outputs(out, batch_index=0)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            target = reconstruct_target_grid(
                ex["target_cells"], int(ex["target_height"]), int(ex["target_width"])
            )
            exacts.append(exact_match(pred, target))
            pixels.append(pixel_accuracy(pred, target))
            shapes.append(shape_accuracy(pred, target))
        metrics = aggregate(exacts, pixels, shapes, latencies)
        results[str(depth)] = metrics.to_dict()
        logger.info(
            "depth=%d | exact=%.3f pixel=%.3f shape=%.3f mean_latency=%.1fms n=%d",
            depth, metrics.exact_match, metrics.pixel_accuracy, metrics.shape_accuracy,
            metrics.mean_latency_ms, metrics.num_tasks,
        )
    return results


def run_evaluation(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    split: str,
    depths: list[int],
    out_path: str | Path | None = None,
    limit: int | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(ckpt)
    device = resolve_device()
    dataset = _build_split_dataset(config, split)
    depth_metrics = evaluate_depths(model, dataset, depths, device, limit=limit)

    report = {
        "model_id": model_id or config.get("model", {}).get("name", "model"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": file_sha256(checkpoint_path),
        "dataset_hash": ckpt.get("dataset_hash") or _dataset_hash(),
        "config_hash": dict_sha256(config),
        "split": split,
        "device": device,
        "depths": depth_metrics,
    }
    if out_path:
        write_json(out_path, report)
        logger.info("Wrote eval report -> %s", out_path)
    return report


def main() -> int:
    from packages.common.config import load_yaml_config

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint across recursion depths")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train_holdout")
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    report = run_evaluation(
        args.checkpoint, config, args.split, args.depths, out_path=args.out, limit=args.limit
    )
    print(f"OK: evaluated {report['model_id']} on {args.split} at depths {args.depths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
