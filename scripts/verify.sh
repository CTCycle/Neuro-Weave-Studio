#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/backend"
uv sync --extra test
uv run ruff check .
uv run pytest
cd "$ROOT/frontend"
npm install
npm run lint
npm run test
npm run build
