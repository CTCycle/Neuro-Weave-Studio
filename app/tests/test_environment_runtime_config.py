# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from common.path import (
    CACHE_ROOT,
    DATABASE_PATH,
    DEFAULT_RESOURCE_ROOT,
    ENV_EXAMPLE_FILE,
    FRONTEND_DIST_ROOT,
    PYTHON_BYTECODE_CACHE_ROOT,
    REPOSITORY_ROOT,
    RESOURCE_ROOT,
    RESOURCE_ROOT_ENV,
    RUNTIME_DISCOVERY_CACHE_PATH,
    RUNTIME_ROOT,
    resolve_resource_root,
)
from configurations.environment import ensure_environment_file
from configurations.runtime_config import RuntimeConfiguration, load_runtime_configuration
from configurations.settings import Settings

# Consolidated from test_environment_contract.py

###############################################################################
def test_concurrent_first_start_creates_one_exact_environment_file(
    tmp_path: Path,
) -> None:
    template = tmp_path / ".env.example"
    environment_file = tmp_path / "settings" / ".env"
    expected = b"FASTAPI_HOST=127.0.0.1\n"
    template.write_bytes(expected)

    def ensure(_: int) -> Path:
        return ensure_environment_file(environment_file, template)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(ensure, range(8)))

    assert results == [environment_file] * 8
    assert environment_file.read_bytes() == expected

###############################################################################
def test_existing_environment_file_is_never_overwritten(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    environment_file = tmp_path / ".env"
    template.write_text(
        "FASTAPI_HOST=127.0.0.1\n",
        encoding="utf-8",
    )
    original = b"FASTAPI_HOST=127.0.0.1\n"
    environment_file.write_bytes(original)

    ensure_environment_file(environment_file, template)

    assert environment_file.read_bytes() == original

###############################################################################
def test_environment_template_contains_deployment_settings_not_database_settings() -> None:
    template = ENV_EXAMPLE_FILE.read_text(encoding="utf-8")

    assert "DATABASE_" not in template
    assert "NEURO_WEAVE_API_BASE_URL=/api" in template
    assert "NEURO_WEAVE_CORS_ORIGINS=" in template
    assert "NEURO_WEAVE_LOG_LEVEL" not in template

# Consolidated from test_cache_contract.py

###############################################################################
def test_repository_cache_paths_share_one_canonical_root() -> None:
    assert CACHE_ROOT == REPOSITORY_ROOT / "runtimes" / "cache"
    assert sys.pycache_prefix is not None
    assert Path(sys.pycache_prefix).resolve() == PYTHON_BYTECODE_CACHE_ROOT.resolve()
    assert FRONTEND_DIST_ROOT.parent == CACHE_ROOT
    assert RUNTIME_DISCOVERY_CACHE_PATH.parent == CACHE_ROOT

###############################################################################
def test_resource_root_defaults_to_repository_data() -> None:
    expected = REPOSITORY_ROOT / "data"

    assert expected == DEFAULT_RESOURCE_ROOT
    assert expected == RESOURCE_ROOT
    assert expected / "database.db" == DATABASE_PATH
    assert expected / "runtime" == RUNTIME_ROOT

###############################################################################
def test_resource_root_accepts_repository_relative_and_absolute_paths(tmp_path: Path) -> None:
    assert resolve_resource_root("alternate-resources") == REPOSITORY_ROOT / "alternate-resources"
    assert resolve_resource_root(str(tmp_path / "external-resources")) == (
        tmp_path / "external-resources"
    ).resolve()

###############################################################################
def test_resource_root_override_uses_inherited_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "configured-resources"
    monkeypatch.setenv(RESOURCE_ROOT_ENV, str(override))

    assert resolve_resource_root() == override.resolve()

###############################################################################
def test_runtime_discovery_defaults_to_the_canonical_cache_root() -> None:
    settings = Settings()

    assert settings.runtime_discovery_path == RUNTIME_DISCOVERY_CACHE_PATH.resolve()

###############################################################################
def test_runtime_discovery_cannot_be_redirected_to_persistent_application_data() -> None:
    persistent_path = Path("data/runtime/environment-discovery.json")

    with pytest.raises(ValidationError, match="canonical cache root"):
        Settings(runtime_discovery_path=persistent_path)

# Consolidated from test_runtime_configuration.py

def test_process_environment_overrides_values_from_env_file(tmp_path: Path) -> None:
    environment_file = tmp_path / ".env"
    environment_file.write_text(
        "NEURO_WEAVE_API_BASE_URL=/from-file\n"
        "NEURO_WEAVE_CORS_ORIGINS=https://file.example\n",
        encoding="utf-8",
    )

    configuration = load_runtime_configuration(
        {
            "NEURO_WEAVE_API_BASE_URL": "https://api.example.test/v2/",
            "NEURO_WEAVE_CORS_ORIGINS": "https://ui.example.test/",
        },
        environment_file,
    )

    assert configuration.api_base_url == "https://api.example.test/v2"
    assert configuration.cors_origins == "https://ui.example.test"


def test_empty_process_api_base_url_overrides_env_file_with_default(tmp_path: Path) -> None:
    environment_file = tmp_path / ".env"
    environment_file.write_text("NEURO_WEAVE_API_BASE_URL=/from-file\n", encoding="utf-8")

    configuration = load_runtime_configuration(
        {"NEURO_WEAVE_API_BASE_URL": ""},
        environment_file,
    )

    assert configuration.api_base_url == "/api"


@pytest.mark.parametrize(
    "api_base_url",
    [
        "javascript:alert(1)",
        "//api.example.test",
        "/api?token=unsafe",
        "https://api.example.test/path#fragment",
        "https://api.example.test:70000",
    ],
)
def test_runtime_configuration_rejects_invalid_api_base_urls(api_base_url: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfiguration(api_base_url=api_base_url)


@pytest.mark.parametrize(
    "cors_origins",
    [
        "",
        "https://ui.example.test/path",
        "file:///tmp",
        "https://user:secret@ui.example.test",
        "https://ui.example.test:70000",
        "https://ui.example.test?",
        "https://ui.example.test#",
        "https://ui example.test",
    ],
)
def test_runtime_configuration_rejects_invalid_cors_origins(cors_origins: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfiguration(cors_origins=cors_origins)


def test_runtime_configuration_endpoint_is_uncached_and_available_without_database(
    tmp_path: Path,
    app_factory: Callable[..., FastAPI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEURO_WEAVE_API_BASE_URL", "/configured/api")
    unavailable_database = tmp_path / "database-is-a-directory"
    unavailable_database.mkdir()

    with TestClient(app_factory(database_path=unavailable_database)) as client:
        response = client.get("/runtime-config.json")

    assert response.status_code == 200
    assert response.json() == {"api_base_url": "/configured/api"}
    assert response.headers["cache-control"] == "no-store"


def test_cors_preflight_uses_process_configuration(
    app_factory: Callable[..., FastAPI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEURO_WEAVE_CORS_ORIGINS", "https://allowed.example")

    with TestClient(app_factory()) as client:
        allowed = client.options(
            "/api/metadata",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        denied = client.options(
            "/api/metadata",
            headers={
                "Origin": "https://denied.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"
    assert denied.status_code == 400


def test_backend_serves_built_frontend_on_the_api_origin(
    tmp_path: Path,
    app_factory: Callable[..., FastAPI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist = tmp_path / "frontend-dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<app-root></app-root>", encoding="utf-8")
    (dist / "assets" / "main.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr("common.path.FRONTEND_DIST_ROOT", dist)
    monkeypatch.setattr("common.path.FRONTEND_ASSETS_ROOT", dist / "assets")

    with TestClient(app_factory()) as client:
        assert "<app-root>" in client.get("/").text
        assert "<app-root>" in client.get("/models/some-id").text
        assert client.get("/assets/main.js").text == "console.log(1)"
        assert client.get("/health").status_code == 200
        assert client.get("/main-stale.js").status_code == 404
        assert client.get("/openapi.json").status_code == 200
        missing_api = client.get("/api/does-not-exist")

    assert missing_api.status_code == 404
    assert missing_api.json() == {"detail": "API route not found"}
