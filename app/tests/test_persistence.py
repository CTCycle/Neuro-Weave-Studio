# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from typing import Any, cast
from uuid import UUID

import pytest
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from configurations.database_config import DatabaseConfig
from configurations.settings import DatabaseSettings
from domain.enums import ExecutionBackend, ValidationStatus
from domain.graph import ModelGraph, NodeInstance, Position
from repositories import initializer
from repositories.application_settings import ApplicationSettingsRepository
from repositories.backend import build_session_factory
from repositories.base import Base
from repositories.initializer import HEAD_REVISION, initialize_database
from repositories.models import (
    ApplicationSettingsRecord,
    GeneratedCodeSnapshotRecord,
    ProjectEdgeRecord,
    ProjectGraphSettingsRecord,
    ProjectRecord,
    ProjectRevisionRecord,
)
from repositories.session import RepositoryContext
from services.database_configuration_service import DatabaseConfigurationService
from services.runtime_controller import RuntimeController

# Consolidated from test_database_configuration.py
# Copyright Â© 2026 Thomas Virdis
# Licensed under the MIT License.






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
        saved = primary.patch("/api/settings", json={"log_level": "DEBUG"})
        assert saved.status_code == 200, saved.text
        preference = {
            "api": "keras",
            "backend": "pytorch",
            "device_id": "cpu",
            "memory_policy": {"kind": "default"},
            "precision": "float32",
        }
        normalized_preference = {
            **preference,
            "memory_policy": {"kind": "default", "limit_mb": None, "fraction": None},
        }
        selected = primary.patch(
            "/api/settings",
            json={
                "runtime_preferences": {"pytorch": preference},
                "runtime_preference_backend": "pytorch",
            },
        )
        assert selected.status_code == 200, selected.text
        partial = primary.patch("/api/settings", json={"ml_job_history_limit": 101})
        assert partial.status_code == 200, partial.text
        assert partial.json()["runtime_preferences"]["pytorch"] == normalized_preference
        assert partial.json()["runtime_preference_backend"] == "pytorch"

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
            changed = secondary.patch("/api/settings", json={"log_level": "ERROR"})
            assert changed.status_code == 200, changed.text

        switched = primary.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(secondary_database)},
        )
        assert switched.status_code == 200, switched.text
        assert switched.json()["settings"]["log_level"] == "ERROR"
        assert switched.json()["settings"]["runtime_preferences"] == {}
        assert primary.get("/api/settings").json()["log_level"] == "ERROR"
        assert primary.get("/api/projects").json()["projects"] == []

        switched_back = primary.put(
            "/api/settings/database",
            json={"backend": "sqlite", "sqlite_path": str(primary_database)},
        )
        assert switched_back.status_code == 200, switched_back.text
        assert switched_back.json()["settings"]["log_level"] == "DEBUG"
        assert primary.get("/api/settings").json()["log_level"] == "DEBUG"
        assert primary.get("/api/settings").json()["runtime_preferences"]["pytorch"] == normalized_preference

    with TestClient(app_factory(database_path=primary_database)) as restarted:
        restored_settings = restarted.get("/api/settings").json()
        assert restored_settings["runtime_preferences"]["pytorch"] == normalized_preference
        assert restored_settings["runtime_preference_backend"] == "pytorch"

    persisted_configuration = json.loads(primary_config.read_text(encoding="utf-8"))
    assert persisted_configuration == {
        "backend": "sqlite",
        "connect_timeout_seconds": 30,
        "schema_version": 1,
        "sqlite_path": str(primary_database.resolve()),
    }

###############################################################################
def test_invalid_runtime_preferences_do_not_change_persisted_settings(
    client: TestClient,
) -> None:
    initial = client.get("/api/settings").json()
    invalid = client.patch(
        "/api/settings",
        json={
            "runtime_preferences": {
                "tensorflow": {
                    "api": "keras",
                    "backend": "pytorch",
                    "device_id": "cpu",
                    "memory_policy": {"kind": "default"},
                    "precision": "float32",
                }
            },
            "runtime_preference_backend": "tensorflow",
        },
    )
    assert invalid.status_code == 422
    assert client.get("/api/settings").json() == initial

###############################################################################
def test_settings_rejects_retired_database_cors_value_and_cors_patches(
    client: TestClient,
) -> None:
    assert client.patch("/api/settings", json={"cors_origins": "https://ui.example"}).status_code == 422
    application = cast(FastAPI, client.app)
    context = application.state.repository_context
    with context.new_unit_of_work() as uow:
        assert uow.session is not None
        record = uow.session.get(ApplicationSettingsRecord, 1)
        assert record is not None
        record.settings_json = {
            **record.settings_json,
            "cors_origins": "https://old-database-value.example",
        }
        uow.commit()

    with context.new_unit_of_work() as uow:
        assert uow.session is not None
        with pytest.raises(ValidationError, match="cors_origins"):
            ApplicationSettingsRepository(uow.session).get()

###############################################################################
def test_settings_write_failure_restores_runtime_configuration(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = cast(FastAPI, client.app)
    controller = cast(RuntimeController, application.state.runtime_controller)
    original_apply = controller.apply_settings
    applied_log_levels: list[str] = []

    def record_application(settings: Any) -> None:
        applied_log_levels.append(settings.log_level)
        original_apply(settings)

    def fail_save(_repository: ApplicationSettingsRepository, _settings: Any) -> None:
        raise RuntimeError("injected settings write failure")

    monkeypatch.setattr(controller, "apply_settings", record_application)
    monkeypatch.setattr(ApplicationSettingsRepository, "save", fail_save)

    response = client.patch("/api/settings", json={"log_level": "DEBUG"})

    assert response.status_code == 500
    assert applied_log_levels == ["DEBUG", "INFO"]
    assert application.state.settings.log_level == "INFO"
    assert client.get("/api/settings").json()["log_level"] == "INFO"

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

# Consolidated from test_database_contract.py

###############################################################################
def settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(sqlite_path=path)

###############################################################################
def context(path: Path) -> RepositoryContext:
    backend = initialize_database(settings(path))
    return RepositoryContext(
        backend=backend,
        engine=backend.engine,
        session_factory=build_session_factory(backend.engine),
    )

###############################################################################
def test_sqlite_initialization_is_idempotent_and_enforces_foreign_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "database.db"
    first = initialize_database(settings(path))
    with first.engine.connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == HEAD_REVISION
        )
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
    first.engine.dispose()

    second = initialize_database(settings(path))
    with second.engine.connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == HEAD_REVISION
        )
    second.engine.dispose()

###############################################################################
def test_incompatible_unversioned_database_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unversioned.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE unversioned (id INTEGER PRIMARY KEY)"))
    engine.dispose()

    with pytest.raises(RuntimeError, match="no current Alembic revision"):
        initialize_database(settings(path))

    engine = create_engine(f"sqlite:///{path}")
    try:
        assert inspect(engine).get_table_names() == ["unversioned"]
    finally:
        engine.dispose()

###############################################################################
def test_unit_of_work_rolls_back_partial_project(tmp_path: Path) -> None:
    repository = context(tmp_path / "rollback.db")
    project_id: UUID | None = None
    try:
        with (
            pytest.raises(RuntimeError, match="force rollback"),
            repository.new_unit_of_work() as uow,
        ):
            metadata = uow.projects.create(
                name="Rolled back",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=datetime.now(UTC),
                validation_status=ValidationStatus.VALID,
            )
            project_id = metadata.id
            raise RuntimeError("force rollback")

        with repository.new_unit_of_work() as uow:
            assert project_id is not None
            assert uow.projects.get_metadata(project_id) is None
    finally:
        repository.dispose()

###############################################################################
def test_project_delete_cascades_revision_graph_and_generated_source(
    tmp_path: Path,
) -> None:
    repository = context(tmp_path / "cascade.db")
    now = datetime.now(UTC)
    try:
        with repository.new_unit_of_work() as uow:
            project = uow.projects.create(
                name="Cascade",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            graph = ModelGraph()
            uow.graphs.replace(project.id, graph)
            uow.revisions.append(
                project.id,
                revision=1,
                timestamp=now,
            )
            uow.code_snapshots.upsert(
                project.id,
                revision=1,
                source_code="code",
                timestamp=now,
            )
            uow.commit()

        with repository.new_unit_of_work() as uow:
            assert uow.projects.delete(project.id)
            uow.commit()

        with Session(repository.engine) as session:
            assert session.scalar(select(ProjectRecord)) is None
            assert session.scalar(select(ProjectGraphSettingsRecord)) is None
            assert session.scalar(select(ProjectRevisionRecord)) is None
            assert session.scalar(select(GeneratedCodeSnapshotRecord)) is None
    finally:
        repository.dispose()

###############################################################################
def test_cross_project_edge_reference_is_rejected(tmp_path: Path) -> None:
    repository = context(tmp_path / "cross-project.db")
    now = datetime.now(UTC)
    try:
        with repository.new_unit_of_work() as uow:
            first = uow.projects.create(
                name="First",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            second = uow.projects.create(
                name="Second",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            first_graph = ModelGraph(
                nodes=[
                    NodeInstance(
                        id="source",
                        definition_id="input.tabular",
                        position=Position(x=0, y=0),
                    )
                ]
            )
            second_graph = ModelGraph(
                nodes=[
                    NodeInstance(
                        id="target",
                        definition_id="output.classification",
                        position=Position(x=0, y=0),
                    )
                ]
            )
            uow.graphs.replace(first.id, first_graph)
            uow.graphs.replace(second.id, second_graph)
            uow.revisions.append(
                first.id,
                revision=1,
                timestamp=now,
            )
            uow.revisions.append(
                second.id,
                revision=1,
                timestamp=now,
            )
            uow.commit()

        with pytest.raises(IntegrityError), Session(repository.engine) as session:
            session.add(
                ProjectEdgeRecord(
                    project_id=str(first.id),
                    edge_id="cross",
                    ordinal=0,
                    source_node_id="source",
                    source_port_id="output",
                    target_node_id="target",
                    target_port_id="input",
                )
            )
            session.commit()
    finally:
        repository.dispose()

# Consolidated from test_migrations.py

OLD_HEAD_REVISION = "20261021_drop_write_only_persistence_columns"

###############################################################################
def _settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(sqlite_path=path, connect_timeout=10)

###############################################################################
def test_fresh_baseline_matches_metadata_and_reinitializes_idempotently(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.db"

    backend = initialize_database(_settings(path))
    backend.engine.dispose()
    backend = initialize_database(_settings(path))
    try:
        with backend.engine.connect() as connection:
            inspector = inspect(connection)
            assert set(inspector.get_table_names()) == set(Base.metadata.tables) | {
                "alembic_version"
            }
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
                == HEAD_REVISION
            )
            assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1

            for table in Base.metadata.sorted_tables:
                assert {column["name"] for column in inspector.get_columns(table.name)} == {
                    column.name for column in table.columns
                }
                assert {index["name"] for index in inspector.get_indexes(table.name)} == {
                    index.name for index in table.indexes
                }
                expected_foreign_keys = {
                    (
                        tuple(element.parent.name for element in constraint.elements),
                        constraint.elements[0].column.table.name,
                        tuple(element.column.name for element in constraint.elements),
                        constraint.ondelete,
                    )
                    for constraint in table.foreign_key_constraints
                }
                actual_foreign_keys = {
                    (
                        tuple(foreign_key["constrained_columns"]),
                        foreign_key["referred_table"],
                        tuple(foreign_key["referred_columns"]),
                        (foreign_key.get("options") or {}).get("ondelete"),
                    )
                    for foreign_key in inspector.get_foreign_keys(table.name)
                }
                assert actual_foreign_keys == expected_foreign_keys
    finally:
        backend.engine.dispose()

###############################################################################
def test_concurrent_initializers_serialize_on_one_database(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.db"
    barrier = Barrier(2)

    def initialize() -> str:
        barrier.wait()
        backend = initialize_database(_settings(path))
        try:
            with backend.engine.connect() as connection:
                return str(
                    connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
                )
        finally:
            backend.engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize) for _ in range(2)]
        results = [future.result() for future in futures]

    assert results == [HEAD_REVISION, HEAD_REVISION]
    engine = create_engine(f"sqlite:///{path}")
    try:
        assert "alembic_version" in inspect(engine).get_table_names()
        assert inspect(engine).get_table_names().count("projects") == 1
    finally:
        engine.dispose()

###############################################################################
def test_failed_migration_rolls_back_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "rollback.db"

    def fail_upgrade(config: object, _revision: str) -> None:
        connection = cast(Connection, cast(Config, config).attributes["connection"])
        connection.execute(text("CREATE TABLE migration_failure_marker (id INTEGER)"))
        raise RuntimeError("forced migration failure")

    monkeypatch.setattr(initializer.command, "upgrade", fail_upgrade)
    with pytest.raises(RuntimeError, match="forced migration failure"):
        initialize_database(_settings(path))

    engine = create_engine(f"sqlite:///{path}")
    try:
        assert inspect(engine).get_table_names() == []
    finally:
        engine.dispose()

    monkeypatch.undo()
    backend = initialize_database(_settings(path))
    try:
        with backend.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
                == HEAD_REVISION
            )
    finally:
        backend.engine.dispose()

###############################################################################
def test_old_revision_is_rejected_without_mutating_database(tmp_path: Path) -> None:
    path = tmp_path / "old-revision.db"
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            connection.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
            )
            connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
                {"revision": OLD_HEAD_REVISION},
            )
            connection.execute(text("CREATE TABLE old_project_data (value TEXT NOT NULL)"))
            connection.execute(
                text("INSERT INTO old_project_data (value) VALUES ('preserve')")
            )

        with pytest.raises(RuntimeError, match="unknown Alembic revision"):
            initialize_database(_settings(path))

        with engine.connect() as connection:
            assert set(inspect(connection).get_table_names()) == {
                "alembic_version",
                "old_project_data",
            }
            assert connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one() == OLD_HEAD_REVISION
            assert connection.execute(
                text("SELECT value FROM old_project_data")
            ).scalar_one() == "preserve"
    finally:
        engine.dispose()

###############################################################################
def test_unversioned_database_is_rejected_without_mutating_database(tmp_path: Path) -> None:
    path = tmp_path / "unversioned.db"
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE projects (id TEXT PRIMARY KEY)"))
            connection.execute(text("INSERT INTO projects (id) VALUES ('preserve')"))

        with pytest.raises(RuntimeError, match="no current Alembic revision"):
            initialize_database(_settings(path))

        with engine.connect() as connection:
            assert inspect(connection).get_table_names() == ["projects"]
            assert connection.execute(text("SELECT id FROM projects")).scalar_one() == "preserve"
    finally:
        engine.dispose()
