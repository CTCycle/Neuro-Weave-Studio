# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import create_app
from common.path import CACHE_ROOT
from configurations.database_config import (
    APPLICATION_ENV_KEYS,
    DATABASE_ENV_KEYS,
    DatabaseConfig,
)
from services.database_configuration_service import DatabaseConfigurationService
from services.runtime_controller import RuntimeController


###############################################################################
def test_database_configuration_persists_and_switch_loads_isolated_settings(
    tmp_path: Path, app_factory: Callable[..., FastAPI]
) -> None:
    primary_database = tmp_path / "primary.db"
    secondary_database = tmp_path / "secondary.db"
    primary_config = tmp_path / "database-config.json"
    secondary_config = tmp_path / "secondary-config.json"

    with TestClient(app_factory(database_path=primary_database)) as primary:
        current = primary.get("/api/settings")
        assert current.status_code == 200
        primary_settings = current.json()
        primary_settings["log_level"] = "DEBUG"
        saved = primary.put("/api/settings", json=primary_settings)
        assert saved.status_code == 200, saved.text

        connection_test = primary.post(
            "/api/settings/database/test",
            json={"backend": "sqlite", "sqlite_path": str(primary_database)},
        )
        assert connection_test.status_code == 200, connection_test.text
        assert connection_test.json()["success"] is True

        with TestClient(
            app_factory(
                database_path=secondary_database,
                config_path=secondary_config,
            )
        ) as secondary:
            secondary_settings = secondary.get("/api/settings").json()
            secondary_settings["log_level"] = "ERROR"
            changed = secondary.put("/api/settings", json=secondary_settings)
            assert changed.status_code == 200, changed.text

        switched = primary.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(secondary_database)},
        )
        assert switched.status_code == 200, switched.text
        assert switched.json()["settings"]["log_level"] == "ERROR"
        assert primary.get("/api/settings").json()["log_level"] == "ERROR"
        assert primary.get("/api/projects").json()["projects"] == []

        switched_back = primary.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(primary_database)},
        )
        assert switched_back.status_code == 200, switched_back.text
        assert switched_back.json()["settings"]["log_level"] == "DEBUG"
        assert primary.get("/api/settings").json()["log_level"] == "DEBUG"

    persisted_configuration = json.loads(primary_config.read_text(encoding="utf-8"))
    assert persisted_configuration == {
        "backend": "sqlite",
        "connect_timeout_seconds": 30,
        "schema_version": 1,
        "sqlite_path": str(primary_database.resolve()),
    }


###############################################################################
def test_connection_failure_rolls_back_and_unavailable_database_can_be_repaired(
    tmp_path: Path, app_factory: Callable[..., FastAPI]
) -> None:
    active_database = tmp_path / "active.db"
    invalid_database = tmp_path / "is-a-directory"
    invalid_database.mkdir()

    with TestClient(app_factory(database_path=active_database)) as client:
        failed_test = client.post(
            "/api/settings/database/test",
            json={"backend": "sqlite", "sqlite_path": str(invalid_database)},
        )
        assert failed_test.status_code == 422
        failed_switch = client.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(invalid_database)},
        )
        assert failed_switch.status_code == 422
        status = client.get("/api/settings/database").json()
        assert status["available"] is True
        assert status["sqlite_path"] == str(active_database.resolve())
        assert client.get("/api/settings").status_code == 200

    blocked_path = tmp_path / "unavailable.db"
    blocked_path.mkdir()
    broken_configuration = tmp_path / "broken-config.json"
    broken_configuration.write_text(
        DatabaseConfig(sqlite_path=blocked_path).model_dump_json(indent=2),
        encoding="utf-8",
    )
    replacement_database = tmp_path / "replacement.db"
    with TestClient(
        app_factory(
            database_path=blocked_path,
            config_path=broken_configuration,
        )
    ) as repair_client:
        unavailable = repair_client.get("/api/settings/database").json()
        assert unavailable["available"] is False
        assert repair_client.get("/api/settings").status_code == 503

        tested = repair_client.post(
            "/api/settings/database/test",
            json={"backend": "sqlite", "sqlite_path": str(replacement_database)},
        )
        assert tested.status_code == 200, tested.text
        repaired = repair_client.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(replacement_database)},
        )
        assert repaired.status_code == 200, repaired.text
        assert repair_client.get("/api/settings").status_code == 200


###############################################################################
def test_selected_database_file_configuration_survives_application_restart(
    tmp_path: Path, app_factory: Callable[..., FastAPI]
) -> None:
    primary_database = tmp_path / "primary.db"
    selected_database = tmp_path / "selected.db"

    with TestClient(app_factory(database_path=primary_database)) as first:
        switched = first.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(selected_database)},
        )
        assert switched.status_code == 200, switched.text

    with TestClient(app_factory(database_path=primary_database)) as restarted:
        status = restarted.get("/api/settings/database").json()
        assert status["available"] is True
        assert status["sqlite_path"] == str(selected_database.resolve())
        assert restarted.get("/api/settings").status_code == 200
        restored = restarted.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(primary_database)},
        )
        assert restored.status_code == 200, restored.text


###############################################################################
def test_active_runtime_jobs_block_database_switch(
    tmp_path: Path,
    app_factory: Callable[..., FastAPI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_database = tmp_path / "active.db"
    target_database = tmp_path / "target.db"
    with TestClient(app_factory(database_path=active_database)) as client:
        controller = cast(RuntimeController, cast(FastAPI, client.app).state.runtime_controller)
        monkeypatch.setattr(
            controller,
            "list",
            cast(Any, lambda *, active_only=False: [object()]),
        )

        response = client.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(target_database)},
        )

        assert response.status_code == 409
        assert "active runtime jobs" in response.json()["detail"].lower()
        status = client.get("/api/settings/database").json()
        assert status["sqlite_path"] == str(active_database.resolve())
        assert not target_database.exists()


###############################################################################
def test_switch_gate_drains_requests_and_holds_new_requests_until_activation(
    tmp_path: Path,
) -> None:
    async def exercise_gate() -> None:
        manager = DatabaseConfigurationService(
            tmp_path / "database-config.json",
            sanitize_legacy=False,
        )
        await manager.enter_request()
        activation_entered = asyncio.Event()
        allow_activation_to_finish = asyncio.Event()

        async def activate() -> None:
            async with manager.pause_requests():
                activation_entered.set()
                await allow_activation_to_finish.wait()

        activation = asyncio.create_task(activate())
        await asyncio.sleep(0)
        queued_request = asyncio.create_task(manager.enter_request())
        await asyncio.sleep(0)
        assert manager.switching is True
        assert activation_entered.is_set() is False
        assert queued_request.done() is False

        await manager.leave_request()
        await activation_entered.wait()
        assert queued_request.done() is False
        allow_activation_to_finish.set()
        await activation
        await queued_request
        assert manager.switching is False
        await manager.leave_request()

    asyncio.run(exercise_gate())


###############################################################################
def test_legacy_environment_settings_migrate_once_into_file_and_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "legacy.db"
    config_path = tmp_path / "resources" / "database-config.json"
    environment_file = tmp_path / ".env"
    discovery_path = CACHE_ROOT / "pytest-database-settings" / f"{tmp_path.name}-legacy.json"
    environment_file.write_text(
        "\n".join(
            (
                f"DATABASE_SQLITE_PATH={database}",
                "DATABASE_CONNECT_TIMEOUT=45",
                "NEURO_WEAVE_LOG_LEVEL=DEBUG",
                f"NEURO_WEAVE_LOG_FILE={tmp_path / 'logs' / 'backend.log'}",
                f"NEURO_WEAVE_EXPORT_ROOT={tmp_path / 'exports'}",
                f"NEURO_WEAVE_ML_JOB_ROOT={tmp_path / 'jobs'}",
                f"NEURO_WEAVE_TENSOR_INSPECTION_ROOT={tmp_path / 'inspections'}",
                f"NEURO_WEAVE_RUNTIME_DISCOVERY_PATH={discovery_path}",
                "FASTAPI_PORT=9850",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for key in DATABASE_ENV_KEYS | APPLICATION_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    with TestClient(
        create_app(
            database_config_path=config_path,
            environment_file=environment_file,
        )
    ) as client:
        settings = client.get("/api/settings").json()
        assert settings["log_level"] == "DEBUG"
        assert settings["export_root"] == str((tmp_path / "exports").resolve())
        status = client.get("/api/settings/database").json()
        assert status["connect_timeout_seconds"] == 45

    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    assert set(configuration) == {
        "backend",
        "connect_timeout_seconds",
        "schema_version",
        "sqlite_path",
    }
    assert configuration["sqlite_path"] == str(database.resolve())
    migrated_environment = environment_file.read_text(encoding="utf-8")
    assert "DATABASE_" not in migrated_environment
    assert "NEURO_WEAVE_LOG_LEVEL" not in migrated_environment
    assert "FASTAPI_PORT=9850" in migrated_environment
    assert discovery_path.is_relative_to(CACHE_ROOT)
