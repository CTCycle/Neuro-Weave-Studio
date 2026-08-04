from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
CLIENT_ROOT = APP_ROOT / "client"
FRONTEND_DIST_ROOT = CLIENT_ROOT / "dist"
FRONTEND_ASSETS_ROOT = FRONTEND_DIST_ROOT / "assets"
