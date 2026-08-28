from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from alembic import command
from alembic.config import Config
from configurations.settings import DatabaseSettings
from repositories import initializer
from repositories.initializer import HEAD_REVISION, initialize_database
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection

###############################################################################
def _settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(sqlite_path=path, connect_timeout=10)

###############################################################################
def _migration_config(connection: Connection) -> Config:
    config = Config(str(initializer.ALEMBIC_CONFIG_PATH))
    config.attributes["connection"] = connection
    config.attributes["skip_logging_config"] = True
    return config

###############################################################################
def test_concurrent_initializers_serialize_on_one_database(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.db"
    barrier = Barrier(2)

    def initialize() -> str:
        barrier.wait()
        backend = initialize_database(_settings(path))
        try:
            with backend.engine.connect() as connection:
                return str(connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one())
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
            assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == HEAD_REVISION
    finally:
        backend.engine.dispose()

###############################################################################
def test_canonical_model_configuration_migrates_v2_data_and_downgrades(
    tmp_path: Path,
) -> None:
    path = tmp_path / "configuration-migration.db"
    engine = create_engine(f"sqlite:///{path}")
    project_id = "migration-project"
    snapshot = {
        "schema_version": 2,
        "nodes": [],
        "edges": [],
        "model_configuration": {
            "optimizer": "adam",
            "optimizer_parameters": {"learning_rate": 0.002},
            "losses": {"__default__": "mean_squared_error"},
            "metrics": {"__default__": ["mean_absolute_error"]},
            "default_initializer": "he_normal",
            "default_regularizer": "l2",
            "default_constraint": "none",
            "mixed_precision": True,
            "random_seed": 7,
        },
    }
    try:
        with engine.begin() as connection:
            config = _migration_config(connection)
            command.upgrade(config, "20260821_execution_profile")
            connection.execute(
                text(
                    "INSERT INTO projects (id, name, description, graph_schema_version, "
                    "execution_backend, keras_enabled, current_revision, validation_status, "
                    "created_at, updated_at) VALUES (:id, 'Migration model', '', 2, "
                    "'pytorch', 1, 1, 'unknown', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": project_id},
            )
            connection.execute(
                text(
                    "INSERT INTO project_graph_settings (project_id, optimizer, "
                    "optimizer_parameters_json, losses_json, metrics_json, default_initializer, "
                    "default_regularizer, default_constraint, mixed_precision, random_seed, "
                    "include_comments, include_shape_assertions, preferred_binary_format, "
                    "future_training_configuration_json) VALUES (:project_id, 'adam', "
                    ":optimizer_parameters, :losses, :metrics, 'he_normal', 'l2', 'none', "
                    "1, 7, 1, 1, 'keras', '{}')"
                ),
                {
                    "project_id": project_id,
                    "optimizer_parameters": json.dumps({"learning_rate": 0.002}),
                    "losses": json.dumps({"__default__": "mean_squared_error"}),
                    "metrics": json.dumps({"__default__": ["mean_absolute_error"]}),
                },
            )
            connection.execute(
                text(
                    "INSERT INTO project_revisions (project_id, revision, graph_schema_version, "
                    "execution_backend, keras_enabled, created_at, graph_snapshot_json) "
                    "VALUES (:project_id, 1, 2, 'pytorch', 1, CURRENT_TIMESTAMP, :snapshot)"
                ),
                {"project_id": project_id, "snapshot": json.dumps(snapshot)},
            )
            command.upgrade(config, "head")

            settings_columns = {item["name"] for item in inspect(connection).get_columns("project_graph_settings")}
            assert "model_configuration_json" in settings_columns
            assert "optimizer" not in settings_columns
            migrated = connection.execute(
                text("SELECT model_configuration_json FROM project_graph_settings WHERE project_id = :id"),
                {"id": project_id},
            ).scalar_one()
            migrated = json.loads(migrated) if isinstance(migrated, str) else migrated
            assert migrated["keras"]["optimizer_parameters"] == {"learning_rate": 0.002}
            assert migrated["native"]["pytorch"]["compile_enabled"] is False
            assert connection.execute(text("SELECT graph_schema_version FROM projects WHERE id = :id"), {"id": project_id}).scalar_one() == 3
            migrated_snapshot = connection.execute(
                text("SELECT graph_snapshot_json FROM project_revisions WHERE project_id = :id"),
                {"id": project_id},
            ).scalar_one()
            migrated_snapshot = json.loads(migrated_snapshot) if isinstance(migrated_snapshot, str) else migrated_snapshot
            assert migrated_snapshot["schema_version"] == 3
            assert migrated_snapshot["model_configuration"]["native"]["jax"]["jit_enabled"] is False

            command.downgrade(config, "20260821_execution_profile")
            downgraded_columns = {item["name"] for item in inspect(connection).get_columns("project_graph_settings")}
            assert "model_configuration_json" not in downgraded_columns
            assert "optimizer" in downgraded_columns
            assert connection.execute(text("SELECT graph_schema_version FROM projects WHERE id = :id"), {"id": project_id}).scalar_one() == 2
            restored_optimizer = connection.execute(
                text("SELECT optimizer FROM project_graph_settings WHERE project_id = :id"),
                {"id": project_id},
            ).scalar_one()
            assert restored_optimizer == "adam"
    finally:
        engine.dispose()
