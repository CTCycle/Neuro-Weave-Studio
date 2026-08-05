from __future__ import annotations

import logging
from pathlib import Path

import sqlalchemy
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from common.constants import DATABASE_SCHEMA_VERSION
from configurations.settings import DatabaseSettings
from repositories import (
    models as _models,  # pyright: ignore[reportUnusedImport]  # noqa: F401 - register ORM tables
)
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
from repositories.models import SchemaMetadataRecord

LOGGER = logging.getLogger(__name__)

###############################################################################
def _create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        metadata = session.get(SchemaMetadataRecord, 1)
        if metadata is None:
            session.add(
                SchemaMetadataRecord(schema_version=DATABASE_SCHEMA_VERSION)
            )
        elif metadata.schema_version != DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported database schema version: {metadata.schema_version}"
            )
        session.commit()


###############################################################################
def _check_schema_readiness(engine: Engine) -> None:
    """Check the canonical schema marker without inspecting or repairing tables."""
    try:
        with engine.connect() as connection:
            version = connection.execute(
                text("SELECT schema_version FROM schema_metadata WHERE id = 1")
            ).scalar_one_or_none()
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "The configured database is unavailable or does not contain the "
            "canonical NeuroWeave schema. Recreate the database explicitly."
        ) from exc
    if version != DATABASE_SCHEMA_VERSION:
        raise RuntimeError(
            "The configured database does not contain the canonical NeuroWeave "
            "schema. Recreate the database explicitly."
        )


###############################################################################
def _sqlite_database_path(settings: DatabaseSettings) -> Path:
    if settings.sqlite_path is None:
        raise ValueError("SQLite path is required for embedded database mode")
    return Path(settings.sqlite_path).resolve()


###############################################################################
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


###############################################################################
def _refuse_mixed_postgres_schema(engine: Engine) -> None:
    managed_tables = set(inspect(engine).get_table_names()).intersection(
        Base.metadata.tables
    )
    if managed_tables:
        raise RuntimeError(
            "The PostgreSQL database contains an older or incomplete NeuroWeave "
            "schema. Re-run initialization with --reset."
        )


###############################################################################
def initialize_postgres_database(
    settings: DatabaseSettings, *, reset: bool = False
) -> PostgresBackend:
    """Explicitly initialize or recreate the configured PostgreSQL schema."""
    if settings.embedded_database:
        raise ValueError(
            "PostgreSQL initialization requires DATABASE_EMBEDDED=false in settings/.env"
        )

    backend = _ensure_postgres_database(settings)
    try:
        if reset:
            Base.metadata.drop_all(backend.engine)
        else:
            try:
                _check_schema_readiness(backend.engine)
                return backend
            except RuntimeError:
                _refuse_mixed_postgres_schema(backend.engine)
        _create_schema(backend.engine)
        _check_schema_readiness(backend.engine)
    except Exception:
        backend.engine.dispose()
        raise
    LOGGER.info("PostgreSQL schema initialization completed")
    return backend


###############################################################################
def initialize_sqlite_database(
    settings: DatabaseSettings, *, reset: bool = False
) -> SQLiteBackend:
    """Create the SQLite schema, optionally rebuilding the local database file."""
    if not settings.embedded_database:
        raise ValueError("SQLite initialization requires DATABASE_EMBEDDED=true")

    db_path = _sqlite_database_path(settings)
    if reset and db_path.exists():
        db_path.unlink()
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
            _check_schema_readiness(backend.engine)
    except Exception:
        backend.engine.dispose()
        raise
    return backend


###############################################################################
def initialize_database(
    settings: DatabaseSettings, *, reset: bool = False
) -> DatabaseBackend:
    """Initialize the configured database through the explicit lifecycle."""
    if settings.embedded_database:
        return initialize_sqlite_database(settings, reset=reset)
    return initialize_postgres_database(settings, reset=reset)


###############################################################################
def ensure_database_ready(settings: DatabaseSettings) -> DatabaseBackend:
    """Open the configured database without modifying an existing schema."""
    if settings.embedded_database:
        return initialize_sqlite_database(settings)

    backend = build_database_backend(settings)
    assert isinstance(backend, PostgresBackend)
    try:
        _check_schema_readiness(backend.engine)
    except Exception:
        backend.engine.dispose()
        raise
    return backend
