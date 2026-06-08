"""Split policy with eval-leakage guards.

- Official ``training`` tasks are deterministically split into ``train`` and
  ``validation`` (a.k.a. train_holdout) using a fixed seed.
- The official ``evaluation`` split is reserved for FINAL evaluation only and is
  never returned for training. A guard raises ``RuntimeError`` if training code
  attempts to load it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from packages.common.logging import get_logger
from packages.data.schema import ARCTask

logger = get_logger(__name__)

SEED = 42

# Logical split names exposed to the rest of the system.
TRAIN = "train"
VALIDATION = "validation"
TRAIN_HOLDOUT = "train_holdout"  # alias for validation, per spec naming
OFFICIAL_TRAINING = "training"
OFFICIAL_EVALUATION = "evaluation"


@dataclass
class SplitResult:
    train: list[ARCTask]
    validation: list[ARCTask]


class EvalLeakageError(RuntimeError):
    """Raised when official evaluation data is requested for training."""


def make_splits(
    tasks: list[ARCTask],
    holdout_ratio: float = 0.15,
    seed: int = SEED,
) -> SplitResult:
    """Deterministically split official *training* tasks into train/validation.

    Evaluation tasks present in ``tasks`` are ignored here (never used for train).
    """
    train_tasks = sorted(
        (t for t in tasks if t.split == OFFICIAL_TRAINING), key=lambda t: t.task_id
    )
    if not train_tasks:
        raise ValueError("No official training tasks found to split")

    rng = random.Random(seed)
    order = list(range(len(train_tasks)))
    rng.shuffle(order)

    n_val = max(1, int(round(len(train_tasks) * holdout_ratio)))
    val_idx = set(order[:n_val])

    train_split = [t for i, t in enumerate(train_tasks) if i not in val_idx]
    val_split = [t for i, t in enumerate(train_tasks) if i in val_idx]

    logger.info(
        "Split %d official training tasks -> %d train / %d validation (seed=%d)",
        len(train_tasks),
        len(train_split),
        len(val_split),
        seed,
    )
    return SplitResult(train=train_split, validation=val_split)


def select_split(
    tasks: list[ARCTask],
    split: str,
    *,
    holdout_ratio: float = 0.15,
    seed: int = SEED,
    allow_official_eval: bool = False,
    for_training: bool = False,
) -> list[ARCTask]:
    """Return tasks for a logical split, enforcing the eval-leakage guard.

    Args:
        split: one of train | validation | train_holdout | training | evaluation
        for_training: if True, requesting the evaluation split raises unless
            ``allow_official_eval`` is also True.
    """
    split = split.lower()

    if split in (OFFICIAL_EVALUATION,):
        if for_training and not allow_official_eval:
            raise EvalLeakageError(
                "Official evaluation split requested for training. This is forbidden "
                "by the leakage policy. Set allow_official_eval=True only for final "
                "evaluation, never for training/architecture tuning."
            )
        return sorted(
            (t for t in tasks if t.split == OFFICIAL_EVALUATION), key=lambda t: t.task_id
        )

    if split == OFFICIAL_TRAINING:
        return sorted(
            (t for t in tasks if t.split == OFFICIAL_TRAINING), key=lambda t: t.task_id
        )

    result = make_splits(tasks, holdout_ratio=holdout_ratio, seed=seed)
    if split == TRAIN:
        return result.train
    if split in (VALIDATION, TRAIN_HOLDOUT):
        return result.validation

    raise ValueError(f"Unknown split: {split!r}")
