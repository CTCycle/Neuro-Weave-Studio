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
NEURO_WEAVE_DATABASE_URL="sqlite:///$QA_DB" NEURO_WEAVE_EXPORT_ROOT="$QA_ROOT/exports" uv run pytest
NEURO_WEAVE_DATABASE_URL="sqlite:///$QA_DB" uv run alembic upgrade head
cd "$ROOT/app/client"
npm ci
npm run lint
npm run test
npm run build
