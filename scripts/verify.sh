#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QA_ROOT="$ROOT/assets/QA"
mkdir -p "$QA_ROOT/exports"
QA_DB="$QA_ROOT/verify.db"
rm -f "$QA_DB"
cd "$ROOT/app/server"
uv sync --frozen --extra test
uv run ruff check .
EMBEDDED_DATABASE="true" DATABASE_SQLITE_PATH="$QA_DB" NEURO_WEAVE_EXPORT_ROOT="$QA_ROOT/exports" uv run python ../scripts/initialize_database.py --drop-existing
EMBEDDED_DATABASE="true" DATABASE_SQLITE_PATH="$QA_DB" NEURO_WEAVE_EXPORT_ROOT="$QA_ROOT/exports" uv run pytest
cd "$ROOT/app/client"
npm ci
npm run lint
npm run test
npm run build
