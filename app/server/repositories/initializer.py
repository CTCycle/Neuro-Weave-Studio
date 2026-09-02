from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.script.revision import ResolutionError
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from configurations.settings import DatabaseSettings
from repositories import (
    models as _models,  # pyright: ignore[reportUnusedImport]  # noqa: F401 - register ORM tables
)
from repositories.backend import SQLiteBackend
from repositories.base import Base
from repositories.migration_lock import database_migration_lock

LOGGER = logging.getLogger(__name__)

ALEMBIC_CONFIG_PATH = Path(__file__).resolve().parents[1] / "alembic.ini"
HEAD_REVISION = "20260902_runtime_batch_shapes"
ALEMBIC_VERSION_TABLE = "alembic_version"


###############################################################################
def _alembic_config(connection: Connection) -> Config:
    config = Config(str(ALEMBIC_CONFIG_PATH))
    config.attributes["connection"] = connection
    config.attributes["skip_logging_config"] = True
    return config


###############################################################################
def _script_directory(config: Config) -> tuple[ScriptDirectory, str]:
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    if len(heads) != 1:
        raise RuntimeError(
            "The migration environment must contain exactly one head; "
            f"found {', '.join(heads) if heads else 'none'}."
        )
    return script, heads[0]


###############################################################################
def _current_heads(connection: Connection) -> tuple[str, ...]:
    migration_context = MigrationContext.configure(connection)
    return tuple(migration_context.get_current_heads())


###############################################################################
def _validate_known_revisions(current_heads: tuple[str, ...], script: ScriptDirectory) -> None:
    if len(current_heads) > 1:
        raise RuntimeError(
            "The database contains multiple Alembic heads. Merge the migration "
            "branches before starting the application."
        )
    for revision in current_heads:
        try:
            script.get_revision(revision)
        except ResolutionError as exc:
            raise RuntimeError(
                f"The database references unknown Alembic revision {revision!r}. "
                "Recreate the database using the current schema."
            ) from exc


###############################################################################
def _run_migrations(connection: Connection) -> None:
    config = _alembic_config(connection)
    script, expected_head = _script_directory(config)
    current_heads = _current_heads(connection)
    _validate_known_revisions(current_heads, script)

    unversioned_tables = frozenset(inspect(connection).get_table_names()) - {ALEMBIC_VERSION_TABLE}
    if not current_heads and unversioned_tables:
        found_tables = ", ".join(sorted(unversioned_tables))
        raise RuntimeError(
            "The database contains tables but no current Alembic revision. "
            "Unsupported or incomplete schemas are not adopted or mutated. "
            f"Tables found: {found_tables}. Back up/export the data before "
            "configuring a new database."
        )

    if current_heads == (expected_head,):
        LOGGER.info("Database schema is already at Alembic head %s", expected_head)
    else:
        LOGGER.info(
            "Initializing or upgrading database schema to Alembic head %s",
            expected_head,
        )
        command.upgrade(config, "head")

    final_heads = _current_heads(connection)
    if final_heads != (expected_head,):
        raise RuntimeError(
            "Database migration completed without reaching the expected Alembic head "
            f"{expected_head}; current heads are {final_heads or 'none'}."
        )
    actual_tables = frozenset(inspect(connection).get_table_names()) - {ALEMBIC_VERSION_TABLE}
    expected_tables = frozenset(Base.metadata.tables)
    if actual_tables != expected_tables:
        unexpected = sorted(actual_tables - expected_tables)
        missing = sorted(expected_tables - actual_tables)
        raise RuntimeError(
            "Database reached the Alembic head but its managed table set does not "
            f"match the application metadata (missing={missing}, unexpected={unexpected})."
        )
    foreign_keys_enabled = connection.execute(text("PRAGMA foreign_keys")).scalar_one()
    if foreign_keys_enabled != 1:
        raise RuntimeError("SQLite foreign-key enforcement is disabled after migration.")
    LOGGER.info("Database schema verified at Alembic head %s", expected_head)


###############################################################################
def initialize_database(settings: DatabaseSettings) -> SQLiteBackend:
    """Create or upgrade the current SQLite schema with no legacy adoption."""

    backend = SQLiteBackend(settings)
    try:
        with (
            database_migration_lock(backend.db_path, settings.connect_timeout),
            backend.engine.begin() as connection,
        ):
            _run_migrations(connection)
    except Exception:
        LOGGER.exception("Database initialization or migration failed; transaction rolled back")
        backend.engine.dispose()
        raise
    return backend
