from __future__ import annotations

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
RESOURCE_ROOT = APP_ROOT / "resources"
DATABASE_PATH = RESOURCE_ROOT / "database.db"
RUNTIME_ROOT = RESOURCE_ROOT / "runtime"
RUNTIME_LOG_ROOT = RUNTIME_ROOT / "logs"
RUNTIME_EXPORT_ROOT = RUNTIME_ROOT / "exports"
RUNTIME_DISCOVERY_CACHE_PATH = CACHE_ROOT / "environment-discovery.json"
FRONTEND_DIST_ROOT = CACHE_ROOT / "frontend-dist"
FRONTEND_ASSETS_ROOT = FRONTEND_DIST_ROOT / "assets"
ENV_FILE = SETTINGS_ROOT / ".env"
ENV_EXAMPLE_FILE = SETTINGS_ROOT / ".env.example"
