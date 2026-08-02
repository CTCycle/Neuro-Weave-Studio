#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'kill 0' EXIT
(
  cd "$ROOT/app/server"
  uv sync --frozen --extra test
  uv run uvicorn neuro_weave.app:app --host 127.0.0.1 --port 8000
) &
(
  cd "$ROOT/app/client"
  npm ci
  npm run dev -- --host 127.0.0.1 --port 5173
) &
wait
