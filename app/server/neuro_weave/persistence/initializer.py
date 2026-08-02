from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import sqlalchemy
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from neuro_weave.config import DatabaseSettings
from neuro_weave.persistence import models as _models  # noqa: F401 - register ORM tables
from neuro_weave.persistence.backend import (
    DatabaseBackend,
    PostgresBackend,
    SQLiteBackend,
    build_database_backend,
    build_postgres_connect_args,
    build_postgres_url,
    validate_postgres_database_name,
)
from neuro_weave.persistence.base import Base

REQUIRED_TABLES = frozenset({"projects", "project_revisions"})
REQUIRED_COLUMNS = {
    "projects": frozenset(
        {
            "id",
            "name",
            "description",
            "created_at",
            "updated_at",
            "schema_version",
            "target_framework",
            "current_revision",
            "validation_status",
            "graph_json",
            "generated_code_json",
        }
    ),
    "project_revisions": frozenset(
        {"id", "project_id", "revision", "created_at", "graph_json", "target_framework"}
    ),
}


def validate_schema(engine: Engine) -> None:
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Could not connect to the configured database. Verify DATABASE_* settings "
            "or run start_on_windows.ps1 -Mode InitializeDatabase."
        ) from exc
    missing = sorted(REQUIRED_TABLES - tables)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Database schema is missing required table(s): {joined}. "
            "Run start_on_windows.ps1 -Mode InitializeDatabase."
        )
    missing_columns = {
        table: sorted(REQUIRED_COLUMNS[table] - {column["name"] for column in inspector.get_columns(table)})
        for table in REQUIRED_TABLES
    }
    missing_columns = {table: columns for table, columns in missing_columns.items() if columns}
    if missing_columns:
        details = "; ".join(
            f"{table}: {', '.join(columns)}" for table, columns in sorted(missing_columns.items())
        )
        raise RuntimeError(
            f"Database schema is missing required column(s) ({details}). "
            "Run start_on_windows.ps1 -Mode InitializeDatabase."
        )


def _sqlite_sidecars(db_path: Path) -> Iterable[Path]:
    yield db_path
    yield Path(f"{db_path}-wal")
    yield Path(f"{db_path}-shm")


def initialize_sqlite_database(
    settings: DatabaseSettings,
    *,
    drop_existing: bool = False,
) -> SQLiteBackend:
    backend = SQLiteBackend(settings)
    db_path = backend.db_path
    if drop_existing:
        backend.engine.dispose()
        for candidate in _sqlite_sidecars(db_path):
            if candidate.is_file():
                candidate.unlink()
        backend = SQLiteBackend(settings)
    Base.metadata.create_all(backend.engine)
    return backend


def ensure_postgres_database(settings: DatabaseSettings) -> PostgresBackend:
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization")
    target_database = validate_postgres_database_name(settings.database_name)
    admin_engine = sqlalchemy.create_engine(
        build_postgres_url(settings, "postgres"),
        echo=False,
        future=True,
        connect_args=build_postgres_connect_args(settings),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    try:
        with admin_engine.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :database_name"),
                {"database_name": target_database},
            ).scalar_one_or_none()
            if not exists:
                connection.exec_driver_sql(
                    f'CREATE DATABASE "{target_database}" WITH ENCODING \'UTF8\' TEMPLATE template0'
                )
    finally:
        admin_engine.dispose()
    return PostgresBackend(settings)


def initialize_postgres_database(
    settings: DatabaseSettings,
    *,
    drop_existing: bool = False,
) -> PostgresBackend:
    backend = ensure_postgres_database(settings)
    if drop_existing:
        Base.metadata.drop_all(backend.engine)
    Base.metadata.create_all(backend.engine)
    return backend


def initialize_database(
    settings: DatabaseSettings,
    *,
    drop_existing: bool = False,
) -> DatabaseBackend:
    if settings.embedded_database:
        return initialize_sqlite_database(settings, drop_existing=drop_existing)
    return initialize_postgres_database(settings, drop_existing=drop_existing)


def ensure_database_ready(settings: DatabaseSettings) -> DatabaseBackend:
    """Build the selected backend and initialize only a missing SQLite file."""
    if settings.embedded_database:
        backend = build_database_backend(settings)
        assert isinstance(backend, SQLiteBackend)
        if not backend.db_path.exists():
            Base.metadata.create_all(backend.engine)
        else:
            try:
                validate_schema(backend.engine)
            except Exception:
                backend.engine.dispose()
                raise
        return backend

    backend = build_database_backend(settings)
    assert isinstance(backend, PostgresBackend)
    try:
        validate_schema(backend.engine)
    except Exception:
        backend.engine.dispose()
        raise
    return backend
