# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
CLIENT_ROOT = APP_ROOT / "client"
SETTINGS_ROOT = REPOSITORY_ROOT / "settings"
CACHE_ROOT = REPOSITORY_ROOT / "runtimes" / "cache"
PYTHON_BYTECODE_CACHE_ROOT = CACHE_ROOT / "python-bytecode"
sys.pycache_prefix = str(PYTHON_BYTECODE_CACHE_ROOT)
PYTEST_CACHE_ROOT = CACHE_ROOT / "pytest"
PYTEST_BASETEMP_ROOT = CACHE_ROOT / "pytest-tmp"
RESOURCE_ROOT_ENV = "NEURO_WEAVE_RESOURCES_DIR"
DEFAULT_RESOURCE_ROOT = REPOSITORY_ROOT / "data"


###############################################################################
def _read_dotenv_value(name: str) -> str | None:
    inherited = os.environ.get(name)
    if inherited is not None and inherited.strip():
        return inherited.strip().strip('"').strip("'")

    environment_file = REPOSITORY_ROOT / "settings" / ".env"
    try:
        lines = environment_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        separator = line.find("=")
        if separator < 1 or line[:separator].strip() != name:
            continue
        value = line[separator + 1 :].strip().strip('"').strip("'")
        return value or None
    return None


###############################################################################
def resolve_resource_root(value: str | None = None) -> Path:
    raw_value = value if value is not None else _read_dotenv_value(RESOURCE_ROOT_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_RESOURCE_ROOT
    path = Path(raw_value.strip()).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


RESOURCE_ROOT = resolve_resource_root()
DATABASE_PATH = RESOURCE_ROOT / "database.db"
RUNTIME_ROOT = RESOURCE_ROOT / "runtime"
RUNTIME_LOG_ROOT = RUNTIME_ROOT / "logs"
RUNTIME_EXPORT_ROOT = RUNTIME_ROOT / "exports"
RUNTIME_DISCOVERY_CACHE_PATH = CACHE_ROOT / "environment-discovery.json"
FRONTEND_DIST_ROOT = CACHE_ROOT / "frontend-dist"
FRONTEND_ASSETS_ROOT = FRONTEND_DIST_ROOT / "assets"
ENV_FILE = SETTINGS_ROOT / ".env"
ENV_EXAMPLE_FILE = SETTINGS_ROOT / ".env.example"
