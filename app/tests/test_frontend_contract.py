from __future__ import annotations

import json
import re
import tomllib

from configurations.settings import REPOSITORY_ROOT

CLIENT_ROOT = REPOSITORY_ROOT / "app" / "client"

###############################################################################
def test_app_client_is_the_only_frontend_source() -> None:
    assert (CLIENT_ROOT / "package.json").is_file()
    assert (CLIENT_ROOT / "package-lock.json").is_file()
    assert (CLIENT_ROOT / "src" / "App.tsx").is_file()
    assert not (REPOSITORY_ROOT / "frontend").exists()

###############################################################################
def test_frontend_build_and_api_contracts_are_canonical() -> None:
    package = json.loads((CLIENT_ROOT / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["build"] == "tsc -b && vite build"
    assert package["scripts"]["test"] == "vitest run"

    vite_config = (CLIENT_ROOT / "vite.config.ts").read_text(encoding="utf-8")
    api_client = (CLIENT_ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    assert "../../settings" in vite_config
    assert "VITE_API_BASE_URL" in vite_config
    assert "VITE_API_" + "ROOT" not in api_client
    assert "VITE_API_BASE_URL" in api_client

###############################################################################
def test_frontend_and_backend_versions_are_consistent_semver() -> None:
    semver_pattern = re.compile(
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
        r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
        r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
    )
    client_root = REPOSITORY_ROOT / "app" / "client"
    server_root = REPOSITORY_ROOT / "app" / "server"

    package_version = json.loads(
        (client_root / "package.json").read_text(encoding="utf-8")
    )["version"]
    lock_version = json.loads(
        (client_root / "package-lock.json").read_text(encoding="utf-8")
    )["version"]
    backend_project = tomllib.loads(
        (server_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    backend_lock = tomllib.loads(
        (server_root / "uv.lock").read_text(encoding="utf-8")
    )
    backend_lock_version = next(
        package["version"]
        for package in backend_lock["package"]
        if package["name"] == "neuro-weave-backend"
    )
    backend_init = (server_root / "__init__.py").read_text(encoding="utf-8")
    backend_runtime = (server_root / "configurations" / "settings.py").read_text(
        encoding="utf-8"
    )
    backend_init_version = re.search(r'__version__\s*=\s*"([^"]+)"', backend_init)
    backend_runtime_version = re.search(
        r'app_version:\s*str\s*=\s*"([^"]+)"', backend_runtime
    )

    versions = [
        package_version,
        lock_version,
        backend_project["project"]["version"],
        backend_lock_version,
        backend_init_version.group(1) if backend_init_version else None,
        backend_runtime_version.group(1) if backend_runtime_version else None,
    ]
    assert all(isinstance(version, str) and semver_pattern.fullmatch(version) for version in versions)
    assert versions == ["1.0.0"] * len(versions)

###############################################################################
def test_backend_points_at_the_canonical_client_build() -> None:
    backend = (REPOSITORY_ROOT / "app" / "server" / "app.py").read_text(encoding="utf-8")
    paths = (REPOSITORY_ROOT / "app" / "server" / "common" / "path.py").read_text(encoding="utf-8")
    assert "FRONTEND_DIST_ROOT" in backend
    assert "StaticFiles" in backend
    assert "CLIENT_ROOT = APP_ROOT / \"client\"" in paths
