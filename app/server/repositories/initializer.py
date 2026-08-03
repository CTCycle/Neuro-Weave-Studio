from __future__ import annotations

import logging
from pathlib import Path

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from configurations.settings import DatabaseSettings
from repositories import models as _models  # noqa: F401 - register ORM tables
from repositories.backend import (
    DatabaseBackend,
    PostgresBackend,
    SQLiteBackend,
    build_database_backend,
    build_postgres_connect_args,
    build_postgres_url,
    validate_postgres_database_name,
)
from repositories.base import Base

LOGGER = logging.getLogger(__name__)


def _create_schema(engine: Engine) -> None:
    """Create the ORM schema for an explicitly initialized database."""
    Base.metadata.create_all(engine)


def _check_application_readiness(engine: Engine, *, postgres: bool = False) -> None:
    """Run one lightweight readiness query without inspecting or repairing schema."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1 FROM projects LIMIT 1"))
    except SQLAlchemyError as exc:
        detail = (
            "Run the launcher's PostgreSQL initialization action."
            if postgres
            else "Existing SQLite files are never rebuilt automatically."
        )
        raise RuntimeError(
            f"The configured database is unavailable or has not been initialized. {detail}"
        ) from exc


def _sqlite_database_path(settings: DatabaseSettings) -> Path:
    if settings.sqlite_path is None:
        raise ValueError("SQLite path is required for embedded database mode")
    return Path(settings.sqlite_path).resolve()


def _ensure_postgres_database(settings: DatabaseSettings) -> PostgresBackend:
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
                LOGGER.info("Created PostgreSQL database")
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "PostgreSQL database creation failed. Verify the server, database settings, "
            "and administrative permissions."
        ) from exc
    finally:
        admin_engine.dispose()

    return PostgresBackend(settings)


def initialize_postgres_database(settings: DatabaseSettings) -> PostgresBackend:
    """Explicitly create the configured PostgreSQL database and ORM schema."""
    if settings.embedded_database:
        raise ValueError(
            "PostgreSQL initialization requires DATABASE_EMBEDDED=false in settings/.env"
        )

    backend = _ensure_postgres_database(settings)
    try:
        _create_schema(backend.engine)
        _check_application_readiness(backend.engine, postgres=True)
    except Exception:
        backend.engine.dispose()
        raise
    LOGGER.info("PostgreSQL schema initialization completed")
    return backend


def initialize_database(settings: DatabaseSettings) -> PostgresBackend:
    """Run the explicit PostgreSQL initialization workflow only."""
    return initialize_postgres_database(settings)


def ensure_database_ready(settings: DatabaseSettings) -> DatabaseBackend:
    """Open the configured database, bootstrapping only a missing SQLite file."""
    if settings.embedded_database:
        db_path = _sqlite_database_path(settings)
        missing = not db_path.exists()
        if missing:
            db_path.parent.mkdir(parents=True, exist_ok=True)

        backend = build_database_backend(settings)
        assert isinstance(backend, SQLiteBackend)
        try:
            if missing:
                _create_schema(backend.engine)
                LOGGER.info("SQLite database initialized")
            else:
                _check_application_readiness(backend.engine, postgres=False)
        except Exception:
            backend.engine.dispose()
            raise
        return backend

    backend = build_database_backend(settings)
    assert isinstance(backend, PostgresBackend)
    try:
        _check_application_readiness(backend.engine, postgres=True)
    except Exception:
        backend.engine.dispose()
        raise
    return backend
