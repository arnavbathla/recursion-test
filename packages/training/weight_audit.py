"""Weight-update audit: prove the optimizer actually changes model weights.

Snapshots the model's state_dict (and a sha256 of it) before and after running a few
real training steps on real ARC data, then reports:
- how many parameter tensors changed,
- the global L2 magnitude of the weight update,
- the per-tensor top movers,
- whether the state_dict hash changed.

CLI:
    python -m packages.training.weight_audit --config configs/smoke.yaml --steps 5
"""

from __future__ import annotations

import argparse
import hashlib
import io
from typing import Any

import torch

from packages.common.config import load_yaml_config, resolve_device
from packages.common.logging import get_logger
from packages.data.dataset import ARCDataset, collate
from packages.data.ingest_arc import ingest_arc
from packages.data.splits import select_split
from packages.model.losses import arc_loss
from packages.model.utils import build_model, set_seed

logger = get_logger(__name__)


def state_dict_sha256(model: torch.nn.Module) -> str:
    buf = io.BytesIO()
    # Sort keys for determinism; save raw tensors.
    sd = {k: v.detach().cpu() for k, v in sorted(model.state_dict().items())}
    torch.save(sd, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def audit_weight_updates(config: dict[str, Any], steps: int = 5, seed: int = 42) -> dict[str, Any]:
    """Run ``steps`` real training steps and measure how the weights changed."""
    set_seed(seed)
    device = resolve_device()
    data_cfg = config.get("data", {})

    tasks, _ = ingest_arc(data_cfg.get("root", "data/raw/ARC-AGI-2"))
    train_tasks = select_split(tasks, "train", for_training=True)
    max_train = data_cfg.get("max_train_tasks") or 16
    dataset = ARCDataset(train_tasks[: int(max_train)], "train")
    if len(dataset) == 0:
        raise RuntimeError("No trainable examples found for weight audit")

    model = build_model(config.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("lr", 3e-4)))

    before = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    hash_before = state_dict_sha256(model)

    bs = int(config.get("training", {}).get("batch_size", 2))
    idx = 0
    model.train()
    for _ in range(steps):
        items = [dataset[(idx + j) % len(dataset)] for j in range(bs)]
        idx += bs
        batch = collate(items)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        out = model(
            tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
            segments=batch["segments"], answer_positions=batch["answer_positions"],
            attention_mask=batch["attention_mask"], recursion_steps=2,
        )
        loss, _ = arc_loss(out, batch)
        loss.backward()
        optimizer.step()

    after = model.state_dict()
    hash_after = state_dict_sha256(model)

    per_tensor: list[tuple[str, float]] = []
    changed = 0
    total_sq = 0.0
    for k, v_before in before.items():
        delta = float((after[k].detach().cpu() - v_before).norm())
        if delta > 0:
            changed += 1
        total_sq += delta**2
        per_tensor.append((k, delta))

    per_tensor.sort(key=lambda kv: kv[1], reverse=True)
    report = {
        "steps": steps,
        "device": device,
        "num_param_tensors": len(before),
        "num_tensors_changed": changed,
        "total_weight_update_l2": round(total_sq**0.5, 6),
        "state_dict_hash_before": hash_before,
        "state_dict_hash_after": hash_after,
        "state_dict_hash_changed": hash_before != hash_after,
        "weights_changed": changed > 0,
        "top_movers": [{"param": k, "delta_l2": round(d, 6)} for k, d in per_tensor[:8]],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit that training updates model weights")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    report = audit_weight_updates(config, steps=args.steps)
    print("=== Weight-update audit ===")
    print(f"steps:                 {report['steps']}")
    print(f"param tensors:         {report['num_param_tensors']}")
    print(f"tensors changed:       {report['num_tensors_changed']}")
    print(f"total weight L2 delta: {report['total_weight_update_l2']}")
    print(f"state_dict hash before {report['state_dict_hash_before'][:16]}")
    print(f"state_dict hash after  {report['state_dict_hash_after'][:16]}")
    print(f"hash changed:          {report['state_dict_hash_changed']}")
    print(f"WEIGHTS CHANGED:       {report['weights_changed']}")
    print("top movers:")
    for m in report["top_movers"]:
        print(f"  {m['param']:<40} {m['delta_l2']}")
    return 0 if report["weights_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
