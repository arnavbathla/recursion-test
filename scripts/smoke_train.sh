#!/usr/bin/env bash
# Smoke training: train a TINY real model on REAL ARC data for a few steps and
# save checkpoints/smoke.pt. For CI/local validation only -- NOT the final model.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNNER="python"
command -v uv >/dev/null 2>&1 && RUNNER="uv run python"

if [ ! -d "data/raw/ARC-AGI-2/data/training" ]; then
  echo "[smoke] ARC data not found. Run scripts/sync_arc_agi_2.sh first." >&2
  exit 1
fi

echo "[smoke] Training tiny model on real ARC data..."
${RUNNER} -m packages.training.train \
  --config configs/smoke.yaml \
  --model-id smoke \
  --out checkpoints/smoke.pt

echo "[smoke] Done. Checkpoint at checkpoints/smoke.pt"
