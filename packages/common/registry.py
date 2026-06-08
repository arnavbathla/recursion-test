"""Model registry: a JSON file of model versions plus DB persistence helpers.

Each record describes a trained checkpoint with enough provenance to reproduce and
audit it: checkpoint hash, config hash, dataset manifest hash, git commit, metrics.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from packages.common.config import get_settings
from packages.common.hashing import dict_sha256, file_sha256, git_commit
from packages.common.logging import get_logger
from packages.common.storage import read_json, write_json

logger = get_logger(__name__)


@dataclass
class ModelRecord:
    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str | None = None
    config_hash: str | None = None
    dataset_manifest_hash: str | None = None
    git_commit: str | None = None
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _registry_path() -> Path:
    return get_settings().registry_path


def load_registry() -> dict[str, ModelRecord]:
    path = _registry_path()
    if not path.exists():
        return {}
    raw = read_json(path)
    records: dict[str, ModelRecord] = {}
    for item in raw.get("models", []):
        records[item["model_id"]] = ModelRecord(**item)
    return records


def save_registry(records: dict[str, ModelRecord]) -> Path:
    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "models": [r.to_dict() for r in records.values()],
    }
    return write_json(_registry_path(), payload)


def register_model(
    model_id: str,
    checkpoint_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    dataset_manifest_hash: str | None = None,
    metrics: dict[str, Any] | None = None,
    metrics_path: str | None = None,
) -> ModelRecord:
    """Create/replace a registry record and persist the registry JSON."""
    ckpt = Path(checkpoint_path)
    record = ModelRecord(
        model_id=model_id,
        checkpoint_path=str(ckpt),
        checkpoint_sha256=file_sha256(ckpt) if ckpt.exists() else None,
        config_hash=dict_sha256(config) if config is not None else None,
        dataset_manifest_hash=dataset_manifest_hash,
        git_commit=git_commit(),
        metrics=metrics,
        metrics_path=metrics_path,
    )
    records = load_registry()
    records[model_id] = record
    save_registry(records)
    logger.info("Registered model %s -> %s", model_id, record.checkpoint_path)
    return record


def get_record(model_id: str) -> ModelRecord | None:
    return load_registry().get(model_id)


def sync_registry_to_db() -> int:
    """Persist all registry records into the ``model_versions`` table. Returns count."""
    from packages.common.db import ModelVersion, init_db, session_scope

    init_db()
    records = load_registry()
    n = 0
    with session_scope() as session:
        for rec in records.values():
            existing = session.get(ModelVersion, rec.model_id)
            if existing is None:
                session.add(
                    ModelVersion(
                        id=rec.model_id,
                        checkpoint_path=rec.checkpoint_path,
                        checkpoint_sha256=rec.checkpoint_sha256,
                        config_hash=rec.config_hash,
                        dataset_hash=rec.dataset_manifest_hash,
                        git_commit=rec.git_commit,
                        metrics_json=rec.metrics,
                    )
                )
            else:
                existing.checkpoint_path = rec.checkpoint_path
                existing.checkpoint_sha256 = rec.checkpoint_sha256
                existing.config_hash = rec.config_hash
                existing.dataset_hash = rec.dataset_manifest_hash
                existing.git_commit = rec.git_commit
                existing.metrics_json = rec.metrics
            n += 1
    return n
