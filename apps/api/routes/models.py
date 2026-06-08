"""Model registry routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from packages.common.config import REPO_ROOT
from packages.common.registry import get_record, load_registry
from packages.common.storage import read_json

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("")
def list_models() -> dict:
    records = load_registry()
    return {
        "models": [
            {
                "model_id": r.model_id,
                "checkpoint_path": r.checkpoint_path,
                "checkpoint_sha256": r.checkpoint_sha256,
                "config_hash": r.config_hash,
                "dataset_manifest_hash": r.dataset_manifest_hash,
                "git_commit": r.git_commit,
                "created_at": r.created_at,
                "metrics_path": r.metrics_path,
            }
            for r in records.values()
        ]
    }


@router.get("/{model_id}")
def get_model(model_id: str) -> dict:
    r = get_record(model_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return r.to_dict()


@router.get("/{model_id}/training")
def get_training_history(model_id: str) -> dict:
    """Return the training history (train_loss, grad_norm, weight-update L2 per epoch).

    Prefers the dedicated training report under reports/, falling back to the history
    embedded in the registry metrics.
    """
    report_path = REPO_ROOT / "reports" / f"{model_id}_training.json"
    if report_path.exists():
        return read_json(report_path)

    record = get_record(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    metrics = record.metrics or {}
    history = metrics.get("history") or []
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No training history recorded for {model_id}",
        )
    total_movement = sum(float(r.get("weight_update_l2", 0.0)) for r in history)
    return {
        "model_id": model_id,
        "best_metric": metrics.get("best_metric"),
        "total_weight_movement_l2": round(total_movement, 6),
        "weights_changed": total_movement > 0.0,
        "history": history,
    }
