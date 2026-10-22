# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from configurations.runtime_config import (
    RuntimeConfiguration,
    load_runtime_configuration,
)


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
