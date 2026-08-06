from __future__ import annotations

import json

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
def test_backend_points_at_the_canonical_client_build() -> None:
    backend = (REPOSITORY_ROOT / "app" / "server" / "app.py").read_text(encoding="utf-8")
    paths = (REPOSITORY_ROOT / "app" / "server" / "common" / "path.py").read_text(encoding="utf-8")
    assert "FRONTEND_DIST_ROOT" in backend
    assert "StaticFiles" in backend
    assert "CLIENT_ROOT = APP_ROOT / \"client\"" in paths
