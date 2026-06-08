"""Config-driven training entrypoint.

Usage:
    python -m packages.training.train --config configs/trm_arc_v1.yaml
    python -m packages.training.train --config configs/smoke.yaml --model-id smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from packages.common.config import REPO_ROOT, load_yaml_config, resolve_device
from packages.common.hashing import dict_sha256
from packages.common.logging import get_logger
from packages.common.registry import register_model
from packages.common.storage import read_json, write_json
from packages.data.dataset import ARCDataset
from packages.data.ingest_arc import MANIFEST_PATH, ingest_arc
from packages.data.splits import select_split
from packages.model.utils import build_model, set_seed
from packages.training.trainer import TrainConfig, Trainer, make_loaders

logger = get_logger(__name__)


def _dataset_hash() -> str | None:
    p = REPO_ROOT / MANIFEST_PATH
    if p.exists():
        return read_json(p).get("dataset_hash")
    return None


def train_from_config(config: dict[str, Any], model_id: str, out_dir: Path) -> dict[str, Any]:
    set_seed(config.get("training", {}).get("seed", 42))
    data_cfg = config.get("data", {})
    train_cfg_raw = config.get("training", {})

    root = data_cfg.get("root", "data/raw/ARC-AGI-2")
    holdout_ratio = data_cfg.get("train_holdout_ratio", 0.15)
    seed = data_cfg.get("seed", 42)
    allow_eval = data_cfg.get("allow_official_eval_for_training", False)

    tasks, _ = ingest_arc(root)

    # Guard: never load the official evaluation split for training.
    train_tasks = select_split(
        tasks, "train", holdout_ratio=holdout_ratio, seed=seed,
        allow_official_eval=allow_eval, for_training=True,
    )
    val_tasks = select_split(
        tasks, "validation", holdout_ratio=holdout_ratio, seed=seed,
        allow_official_eval=allow_eval, for_training=True,
    )

    max_train = data_cfg.get("max_train_tasks")
    max_val = data_cfg.get("max_val_tasks")
    if max_train:
        train_tasks = train_tasks[: int(max_train)]
    if max_val:
        val_tasks = val_tasks[: int(max_val)]

    train_ds = ARCDataset(train_tasks, "train")
    val_ds = ARCDataset(val_tasks, "validation")

    cfg = TrainConfig(
        mode=train_cfg_raw.get("mode", "deep_supervision"),
        epochs=int(train_cfg_raw.get("epochs", 1)),
        batch_size=int(train_cfg_raw.get("batch_size", 8)),
        lr=float(train_cfg_raw.get("lr", 3e-4)),
        weight_decay=float(train_cfg_raw.get("weight_decay", 0.01)),
        grad_clip=float(train_cfg_raw.get("grad_clip", 1.0)),
        mixed_precision=train_cfg_raw.get("mixed_precision", "bf16"),
        recursion_depths=train_cfg_raw.get("recursion_depths", [1, 2, 4, 8, 16, 32]),
        recursion_steps_per_supervision=int(train_cfg_raw.get("recursion_steps_per_supervision", 4)),
        num_supervision_steps=int(train_cfg_raw.get("num_supervision_steps", 8)),
        eval_every_epochs=int(train_cfg_raw.get("eval_every_epochs", 1)),
        checkpoint_every_epochs=int(train_cfg_raw.get("checkpoint_every_epochs", 1)),
        num_workers=int(train_cfg_raw.get("num_workers", 0)),
        max_steps=train_cfg_raw.get("max_steps"),
        seed=int(train_cfg_raw.get("seed", 42)),
        eval_depth=int(train_cfg_raw.get("eval_depth", 16)),
    )

    model = build_model(config.get("model", {}))
    device = resolve_device()
    logger.info("Training %s on device=%s (%d train / %d val examples)",
                model_id, device, len(train_ds), len(val_ds))

    train_loader, val_loader = make_loaders(train_ds, val_ds, cfg.batch_size, cfg.num_workers)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
        full_config=config,
        dataset_hash=_dataset_hash(),
    )
    result = trainer.fit()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Recursive ARC Engine model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--out", default=None, help="checkpoint output path (.pt)")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    model_id = args.model_id or config.get("model", {}).get("name", "model")

    out_path = Path(args.out) if args.out else REPO_ROOT / "checkpoints" / f"{model_id}.pt"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    work_dir = out_path.parent / f".{model_id}_work"

    result = train_from_config(config, model_id, work_dir)

    # Promote the best checkpoint to the canonical output path.
    best = result.get("best_checkpoint") or result.get("last_checkpoint")
    if best is None:
        raise RuntimeError("Training produced no checkpoint")
    import shutil

    shutil.copyfile(best, out_path)
    logger.info("Promoted %s -> %s", best, out_path)

    # Persist a training-history report: per-epoch train_loss, grad_norm, and the
    # weight-update magnitude (direct evidence the optimizer changed the weights).
    history = result.get("history") or []
    total_weight_movement = sum(float(r.get("weight_update_l2", 0.0)) for r in history)
    training_report = {
        "model_id": model_id,
        "config_hash": dict_sha256(config),
        "dataset_hash": _dataset_hash(),
        "best_metric": result.get("best_metric"),
        "total_steps": history[-1]["steps"] if history else 0,
        "total_weight_movement_l2": round(total_weight_movement, 6),
        "weights_changed": total_weight_movement > 0.0,
        "history": history,
    }
    training_report_path = REPO_ROOT / "reports" / f"{model_id}_training.json"
    write_json(training_report_path, training_report)
    logger.info(
        "Training report -> %s (total weight movement L2=%.4f, weights_changed=%s)",
        training_report_path, total_weight_movement, training_report["weights_changed"],
    )

    register_model(
        model_id,
        out_path,
        config=config,
        dataset_manifest_hash=_dataset_hash(),
        metrics={
            "best_metric": result.get("best_metric"),
            "total_weight_movement_l2": training_report["total_weight_movement_l2"],
            "history": history,
        },
    )
    print(f"OK: trained {model_id}. checkpoint={out_path} best_metric={result.get('best_metric')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
