#!/usr/bin/env bash
# Full training on real ARC-AGI-2 data using configs/trm_arc_v1.yaml.
#
# This is GPU-intensive for the configured budget (200 epochs, d_model=256). It runs
# on CPU/MPS too but will be slow. Tune epochs/batch_size in the config for your
# hardware. Produces checkpoints/trm_arc_v1.pt and registers the model.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${1:-configs/trm_arc_v1.yaml}"
MODEL_ID="${2:-trm_arc_v1}"

RUNNER="python"
command -v uv >/dev/null 2>&1 && RUNNER="uv run python"

if [ ! -d "data/raw/ARC-AGI-2/data/training" ]; then
  echo "[full] ARC data not found. Run scripts/sync_arc_agi_2.sh first." >&2
  exit 1
fi

echo "[full] Training ${MODEL_ID} with ${CONFIG}..."
${RUNNER} -m packages.training.train \
  --config "${CONFIG}" \
  --model-id "${MODEL_ID}" \
  --out "checkpoints/${MODEL_ID}.pt"

echo "[full] Done. Checkpoint at checkpoints/${MODEL_ID}.pt"
