from __future__ import annotations

import configparser
import json
import tomllib

from configurations.settings import REPOSITORY_ROOT

CACHE_ROOT = REPOSITORY_ROOT / "assets" / "cache"
SERVER_PROJECT = REPOSITORY_ROOT / "app" / "server" / "pyproject.toml"
CLIENT_ROOT = REPOSITORY_ROOT / "app" / "client"
PYTEST_CONFIG = REPOSITORY_ROOT / "pytest.ini"


###############################################################################
def test_python_development_tools_use_the_central_cache_root() -> None:
    project = tomllib.loads(SERVER_PROJECT.read_text(encoding="utf-8"))
    tool = project["tool"]
    pytest_config = configparser.ConfigParser()
    pytest_config.read(PYTEST_CONFIG, encoding="utf-8")

    assert tool["uv"]["cache-dir"] == "../../assets/cache/uv"
    assert tool["ruff"]["cache-dir"] == "../../assets/cache/ruff"
    assert pytest_config["pytest"]["cache_dir"] == "assets/cache/pytest"
    assert pytest_config["pytest"]["addopts"] == "-q"
    assert pytest_config["pytest"]["pythonpath"] == "app/server"
    conftest = (REPOSITORY_ROOT / "app" / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert '"assets" / "cache" / "pytest-tmp"' in conftest


###############################################################################
def test_frontend_development_tools_use_the_central_cache_root() -> None:
    package = json.loads((CLIENT_ROOT / "package.json").read_text(encoding="utf-8"))
    vite_config = (CLIENT_ROOT / "vite.config.ts").read_text(encoding="utf-8")
    tsconfig_node = (CLIENT_ROOT / "tsconfig.node.json").read_text(encoding="utf-8")
    npmrc = (CLIENT_ROOT / ".npmrc").read_text(encoding="utf-8")

    assert "--cache-location ../../assets/cache/eslint/" in package["scripts"]["lint"]
    assert "path.join(cacheRoot, 'vite')" in vite_config
    assert "path.join(cacheRoot, 'frontend-dist')" in vite_config
    assert "../../assets/cache/typescript/tsconfig.node.tsbuildinfo" in tsconfig_node
    assert "cache=../../assets/cache/npm" in npmrc
    assert CACHE_ROOT.name == "cache"
