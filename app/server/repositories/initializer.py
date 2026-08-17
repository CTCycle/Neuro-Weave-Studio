from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from common.constants import DATABASE_SCHEMA_VERSION
from configurations.settings import DatabaseSettings
from repositories import (
    models as _models,  # pyright: ignore[reportUnusedImport]  # noqa: F401 - register ORM tables
)
from repositories.backend import SQLiteBackend
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
            "The configured SQLite database is unavailable or does not contain "
            "the canonical NeuroWeave schema. Recreate it explicitly."
        ) from exc
    if version != DATABASE_SCHEMA_VERSION:
        raise RuntimeError(
            "The configured SQLite database does not contain the canonical "
            "NeuroWeave schema. Recreate it explicitly."
        )

###############################################################################
def initialize_sqlite_database(
    settings: DatabaseSettings, *, reset: bool = False
) -> SQLiteBackend:
    """Create the SQLite schema, optionally rebuilding the local database file."""
    db_path = Path(settings.sqlite_path).resolve()
    if reset and db_path.exists():
        db_path.unlink()
    missing = not db_path.exists()
    if missing:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    backend = SQLiteBackend(settings)
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
) -> SQLiteBackend:
    """Initialize or recreate the configured SQLite database schema."""
    return initialize_sqlite_database(settings, reset=reset)

###############################################################################
def ensure_database_ready(settings: DatabaseSettings) -> SQLiteBackend:
    """Open SQLite without modifying an existing canonical schema."""
    return initialize_sqlite_database(settings)
