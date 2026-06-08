#!/usr/bin/env bash
# Run the Next.js web UI in dev mode.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/apps/web"

export NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://localhost:8080}"
export API_BASE_URL="${API_BASE_URL:-http://localhost:8080}"

if [ ! -d node_modules ]; then
  echo "[web] Installing dependencies..."
  npm install
fi

exec npm run dev
