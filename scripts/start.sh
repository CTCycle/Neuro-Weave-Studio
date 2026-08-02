#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'kill 0' EXIT
(
  cd "$ROOT/app/server"
  uv sync --extra test
  PYTHONPATH="$ROOT/app/server" uv run uvicorn app:app --reload --port 8000
) &
(
  cd "$ROOT/frontend"
  npm install
  npm run dev
) &
wait
