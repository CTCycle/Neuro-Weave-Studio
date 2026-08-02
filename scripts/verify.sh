#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/app/server"
uv sync --extra test
PYTHONPATH="$ROOT/app/server" uv run ruff check . "$ROOT/app/tests"
PYTHONPATH="$ROOT/app/server" uv run pytest "$ROOT/app/tests"
cd "$ROOT/frontend"
npm install
npm run lint
npm run test
npm run build
