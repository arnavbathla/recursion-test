#!/usr/bin/env bash
# Sync official ARC-AGI-2 public data into data/raw/ and validate it.
#
# This does NOT vendor ARC data into the repo; it fetches it from the official
# source so the dataset stays authoritative and updatable.
set -euo pipefail

REPO_URL="https://github.com/arcprize/ARC-AGI-2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
TARGET="${RAW_DIR}/ARC-AGI-2"

mkdir -p "${RAW_DIR}"

if [ -d "${TARGET}/.git" ]; then
  echo "[sync] ARC-AGI-2 already present. Pulling latest..."
  git -C "${TARGET}" pull --ff-only
else
  echo "[sync] Cloning ARC-AGI-2 from ${REPO_URL}..."
  git clone --depth 1 "${REPO_URL}" "${TARGET}"
fi

TRAIN_DIR="${TARGET}/data/training"
EVAL_DIR="${TARGET}/data/evaluation"

if [ ! -d "${TRAIN_DIR}" ] || [ ! -d "${EVAL_DIR}" ]; then
  echo "[sync] ERROR: expected ${TRAIN_DIR} and ${EVAL_DIR} to exist." >&2
  exit 1
fi

TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
EVAL_COUNT=$(find "${EVAL_DIR}" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')

echo "[sync] training JSON files:   ${TRAIN_COUNT}"
echo "[sync] evaluation JSON files: ${EVAL_COUNT}"

echo "[sync] Running ingestion validator + manifest..."
cd "${ROOT_DIR}"
if command -v uv >/dev/null 2>&1; then
  uv run python -m packages.data.ingest_arc "data/raw/ARC-AGI-2"
else
  python -m packages.data.ingest_arc "data/raw/ARC-AGI-2"
fi

echo "[sync] Done."
