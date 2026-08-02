"""Canonical repository paths shared by backend configuration and tooling."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
SERVER_ROOT = APP_ROOT / "server"
CLIENT_ROOT = APP_ROOT / "client"
SETTINGS_ROOT = REPOSITORY_ROOT / "settings"
EXAMPLES_ROOT = APP_ROOT / "resources" / "examples"
RUNTIME_ROOT = APP_ROOT / "resources" / "runtime"
RUNTIME_LOG_ROOT = RUNTIME_ROOT / "logs"
RUNTIME_EXPORT_ROOT = RUNTIME_ROOT / "exports"
