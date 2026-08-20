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
from repositories.initializer import HEAD_REVISION, initialize_database


###############################################################################
def _settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(sqlite_path=path, connect_timeout=10)

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
