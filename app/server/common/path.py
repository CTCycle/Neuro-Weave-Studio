from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
CLIENT_ROOT = APP_ROOT / "client"
SETTINGS_ROOT = REPOSITORY_ROOT / "settings"
RESOURCE_ROOT = APP_ROOT / "resources"
DATABASE_PATH = RESOURCE_ROOT / "database.db"
RUNTIME_ROOT = RESOURCE_ROOT / "runtime"
RUNTIME_LOG_ROOT = RUNTIME_ROOT / "logs"
RUNTIME_EXPORT_ROOT = RUNTIME_ROOT / "exports"
FRONTEND_DIST_ROOT = CLIENT_ROOT / "dist"
FRONTEND_ASSETS_ROOT = FRONTEND_DIST_ROOT / "assets"
ENV_FILE = SETTINGS_ROOT / ".env"
ENV_EXAMPLE_FILE = SETTINGS_ROOT / ".env.example"
