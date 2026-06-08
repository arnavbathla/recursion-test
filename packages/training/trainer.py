"""Training loop with two modes.

Mode A ("simple"): each step samples a recursion depth from a configured set, runs
one forward pass, and backprops the loss.

Mode B ("deep_supervision"): the TRM-style state-carry loop. The model is run for a
fixed number of supervision steps; after each, loss is accumulated and the latent +
answer states are detached and carried into the next supervision step. The total
loss is backpropagated once.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from packages.common.logging import get_logger
from packages.data.dataset import collate
from packages.data.tokenizer import MAX_GRID
from packages.eval.verifier import exact_match, pixel_accuracy
from packages.inference.decode import decode_outputs
from packages.model.losses import arc_loss
from packages.model.utils import set_seed

logger = get_logger(__name__)


def reconstruct_target_grid(target_cells, th: int, tw: int) -> list[list[int]] | None:
    if th < 0 or tw < 0:
        return None
    h, w = th + 1, tw + 1
    grid = []
    for r in range(h):
        grid.append([int(target_cells[r * MAX_GRID + c]) for c in range(w)])
    return grid


@dataclass
class TrainConfig:
    mode: str = "deep_supervision"
    epochs: int = 1
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"
    recursion_depths: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    recursion_steps_per_supervision: int = 4
    num_supervision_steps: int = 8
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    num_workers: int = 0
    max_steps: int | None = None
    seed: int = 42
    eval_depth: int = 16


@dataclass
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader | None
    cfg: TrainConfig
    device: str
    out_dir: Path
    full_config: dict[str, Any]
    dataset_hash: str | None = None

    def __post_init__(self) -> None:
        set_seed(self.cfg.seed)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.use_bf16 = (
            self.cfg.mixed_precision == "bf16"
            and self.device == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self.best_metric = -1.0
        self.global_step = 0

    def _move(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return out

    def _forward_loss_simple(self, batch: dict) -> torch.Tensor:
        depth = random.choice(self.cfg.recursion_depths)
        outputs = self.model(
            tokens=batch["tokens"],
            rows=batch["rows"],
            cols=batch["cols"],
            segments=batch["segments"],
            answer_positions=batch["answer_positions"],
            attention_mask=batch["attention_mask"],
            recursion_steps=depth,
        )
        loss, _ = arc_loss(outputs, batch)
        return loss

    def _forward_loss_deep_supervision(self, batch: dict) -> torch.Tensor:
        latent_state = None
        answer_state = None
        total_loss = torch.zeros((), device=self.device)
        for _ in range(self.cfg.num_supervision_steps):
            outputs = self.model(
                tokens=batch["tokens"],
                rows=batch["rows"],
                cols=batch["cols"],
                segments=batch["segments"],
                answer_positions=batch["answer_positions"],
                attention_mask=batch["attention_mask"],
                recursion_steps=self.cfg.recursion_steps_per_supervision,
                latent_state=latent_state,
                answer_state=answer_state,
            )
            loss, _ = arc_loss(outputs, batch)
            total_loss = total_loss + loss
            latent_state = outputs.latent_state.detach()
            answer_state = outputs.answer_state.detach()
        return total_loss

    def _grad_global_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return total**0.5

    def train_step(self, batch: dict) -> dict[str, float]:
        """Run one optimizer step. Returns loss + gradient global-norm for logging."""
        self.model.train()
        batch = self._move(batch)
        self.optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_bf16
            else _nullcontext()
        )
        with autocast_ctx:
            if self.cfg.mode == "deep_supervision":
                loss = self._forward_loss_deep_supervision(batch)
            else:
                loss = self._forward_loss_simple(batch)

        loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            # clip_grad_norm_ returns the total grad norm *before* clipping.
            grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip))
        else:
            grad_norm = self._grad_global_norm()
        self.optimizer.step()
        self.global_step += 1
        return {"loss": float(loss.detach()), "grad_norm": grad_norm}

    @torch.no_grad()
    def _param_vector(self) -> torch.Tensor:
        """Flat detached copy of all trainable parameters (for weight-delta tracking)."""
        return torch.cat([p.detach().reshape(-1).float().cpu() for p in self.model.parameters()])

    @torch.no_grad()
    def evaluate(self, depth: int | None = None) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        depth = depth or self.cfg.eval_depth
        n = 0
        n_exact = 0
        pix_sum = 0.0
        for batch in self.val_loader:
            moved = self._move(batch)
            outputs = self.model(
                tokens=moved["tokens"],
                rows=moved["rows"],
                cols=moved["cols"],
                segments=moved["segments"],
                answer_positions=moved["answer_positions"],
                attention_mask=moved["attention_mask"],
                recursion_steps=depth,
            )
            bsz = moved["tokens"].shape[0]
            for i in range(bsz):
                if not bool(batch["has_target"][i]):
                    continue
                target = reconstruct_target_grid(
                    batch["target_cells"][i],
                    int(batch["target_height"][i]),
                    int(batch["target_width"][i]),
                )
                if target is None:
                    continue
                pred = decode_outputs(outputs, batch_index=i)
                n += 1
                if exact_match(pred, target):
                    n_exact += 1
                pix_sum += pixel_accuracy(pred, target)
        if n == 0:
            return {"val_exact_match": 0.0, "val_pixel_accuracy": 0.0, "val_n": 0}
        return {
            "val_exact_match": n_exact / n,
            "val_pixel_accuracy": pix_sum / n,
            "val_n": float(n),
            "val_depth": float(depth),
        }

    def fit(self) -> dict[str, Any]:
        from packages.training.checkpoint import save_checkpoint

        history: list[dict[str, Any]] = []
        last_ckpt: Path | None = None
        best_ckpt: Path | None = None
        stop = False

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_losses: list[float] = []
            epoch_grad_norms: list[float] = []
            params_before = self._param_vector()  # snapshot weights at epoch start
            t0 = time.time()
            for batch in self.train_loader:
                step_metrics = self.train_step(batch)
                epoch_losses.append(step_metrics["loss"])
                epoch_grad_norms.append(step_metrics["grad_norm"])
                if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                    stop = True
                    break

            params_after = self._param_vector()
            # L2 magnitude of the weight change over this epoch: direct evidence that
            # the optimizer actually updated the parameters.
            weight_update_l2 = float((params_after - params_before).norm())
            param_global_norm = float(params_after.norm())

            train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            mean_grad_norm = sum(epoch_grad_norms) / max(1, len(epoch_grad_norms))
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "grad_norm": round(mean_grad_norm, 6),
                "weight_update_l2": round(weight_update_l2, 6),
                "param_global_norm": round(param_global_norm, 6),
                "steps": self.global_step,
                "seconds": round(time.time() - t0, 2),
            }

            do_eval = (epoch % max(1, self.cfg.eval_every_epochs) == 0) or stop
            if do_eval and self.val_loader is not None:
                row.update(self.evaluate())

            logger.info("epoch %d | %s", epoch, row)
            history.append(row)

            do_ckpt = (epoch % max(1, self.cfg.checkpoint_every_epochs) == 0) or stop
            if do_ckpt:
                last_ckpt = save_checkpoint(
                    self.out_dir / "last.pt",
                    self.model,
                    self.full_config,
                    epoch=epoch,
                    step=self.global_step,
                    metrics=row,
                    dataset_hash=self.dataset_hash,
                )

            metric = row.get("val_exact_match")
            if metric is None:
                metric = -train_loss  # fall back to loss-based selection
            if metric > self.best_metric:
                self.best_metric = metric
                best_ckpt = save_checkpoint(
                    self.out_dir / "best.pt",
                    self.model,
                    self.full_config,
                    epoch=epoch,
                    step=self.global_step,
                    metrics=row,
                    dataset_hash=self.dataset_hash,
                )

            if stop:
                break

        return {
            "history": history,
            "best_metric": self.best_metric,
            "last_checkpoint": str(last_ckpt) if last_ckpt else None,
            "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def make_loaders(
    train_ds, val_ds, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader | None]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    return train_loader, val_loader
