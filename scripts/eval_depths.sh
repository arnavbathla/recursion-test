#!/usr/bin/env bash
# Evaluate a checkpoint across recursion depths and render a Markdown report.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="${1:-trm_arc_v1}"
CONFIG="${2:-configs/trm_arc_v1.yaml}"
SPLIT="${3:-train_holdout}"
DEPTHS="${DEPTHS:-1 2 4 8 16 32 64}"
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit ${LIMIT}"

RUNNER="python"
command -v uv >/dev/null 2>&1 && RUNNER="uv run python"

CKPT="checkpoints/${MODEL_ID}.pt"
OUT_JSON="reports/${MODEL_ID}_eval.json"
OUT_MD="reports/${MODEL_ID}_report.md"

echo "[eval] Evaluating ${CKPT} on ${SPLIT} at depths: ${DEPTHS}"
${RUNNER} -m packages.eval.evaluate \
  --checkpoint "${CKPT}" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --depths ${DEPTHS} \
  ${LIMIT_FLAG} \
  --out "${OUT_JSON}"

echo "[eval] Rendering report -> ${OUT_MD}"
${RUNNER} -m packages.eval.report --eval-json "${OUT_JSON}" --out "${OUT_MD}"

echo "[eval] Done."
