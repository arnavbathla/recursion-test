#!/usr/bin/env bash
# Serve the FastAPI backend.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

if command -v uv >/dev/null 2>&1; then
  exec uv run uvicorn apps.api.main:app --host "${HOST}" --port "${PORT}"
else
  exec uvicorn apps.api.main:app --host "${HOST}" --port "${PORT}"
fi
