# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection

from configurations.settings import DatabaseSettings
from repositories import initializer
from repositories.base import Base
from repositories.initializer import HEAD_REVISION, initialize_database

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
