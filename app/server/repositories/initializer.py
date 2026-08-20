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
BASELINE_REVISION = "20260820_v2"
HEAD_REVISION = "20260820_head"
ALEMBIC_VERSION_TABLE = "alembic_version"

_LEGACY_V2_TABLES = frozenset(
    {
        "schema_metadata",
        "projects",
        "project_graph_settings",
        "project_nodes",
        "project_edges",
        "project_revisions",
        "generated_code_snapshots",
    }
)
_LEGACY_V2_COLUMNS = {
    "schema_metadata": frozenset({"id", "schema_version", "initialized_at"}),
    "projects": frozenset(
        {
            "id",
            "name",
            "description",
            "graph_schema_version",
            "code_target",
            "current_revision",
            "validation_status",
            "created_at",
            "updated_at",
        }
    ),
    "project_graph_settings": frozenset(
        {
            "project_id",
            "optimizer",
            "optimizer_parameters_json",
            "losses_json",
            "metrics_json",
            "default_initializer",
            "default_regularizer",
            "default_constraint",
            "mixed_precision",
            "random_seed",
            "include_comments",
            "include_shape_assertions",
            "preferred_binary_format",
            "future_training_configuration_json",
        }
    ),
    "project_nodes": frozenset(
        {
            "project_id",
            "node_id",
            "ordinal",
            "definition_id",
            "definition_version",
            "label",
            "position_x",
            "position_y",
            "parameters_json",
        }
    ),
    "project_edges": frozenset(
        {
            "project_id",
            "edge_id",
            "ordinal",
            "source_node_id",
            "source_port_id",
            "target_node_id",
            "target_port_id",
        }
    ),
    "project_revisions": frozenset(
        {
            "id",
            "project_id",
            "revision",
            "graph_schema_version",
            "code_target",
            "created_at",
            "graph_snapshot_json",
        }
    ),
    "generated_code_snapshots": frozenset(
        {
            "project_id",
            "revision",
            "code_target",
            "source_code",
            "created_at",
            "updated_at",
        }
    ),
}
_LEGACY_V2_PRIMARY_KEYS = {
    "schema_metadata": ("id",),
    "projects": ("id",),
    "project_graph_settings": ("project_id",),
    "project_nodes": ("project_id", "node_id"),
    "project_edges": ("project_id", "edge_id"),
    "project_revisions": ("id",),
    "generated_code_snapshots": ("project_id", "revision", "code_target"),
}

###############################################################################
def _alembic_config(connection: Connection) -> Config:
    config = Config(str(ALEMBIC_CONFIG_PATH))
    config.attributes["connection"] = connection
    config.attributes["skip_logging_config"] = True
    return config

###############################################################################
def _legacy_v2_schema_matches(connection: Connection) -> bool:
    inspector = inspect(connection)
    tables = set(inspector.get_table_names()) - {ALEMBIC_VERSION_TABLE}
    if frozenset(tables) != _LEGACY_V2_TABLES:
        return False
    for table_name, expected_columns in _LEGACY_V2_COLUMNS.items():
        actual_column_records = {
            column["name"]: column for column in inspector.get_columns(table_name)
        }
        if frozenset(actual_column_records) != expected_columns:
            return False
        if table_name == "schema_metadata":
            expected_table = {
                "id": ("SMALLINT", False),
                "schema_version": ("INTEGER", False),
                "initialized_at": ("DATETIME", False),
            }
        else:
            expected_table = {
                column.name: (
                    column.type.compile(dialect=connection.dialect).upper(),
                    bool(column.nullable),
                )
                for column in Base.metadata.tables[table_name].columns
            }
        for column_name, (expected_type, expected_nullable) in expected_table.items():
            actual_column = actual_column_records[column_name]
            if (
                str(actual_column["type"]).upper() != expected_type
                or bool(actual_column["nullable"]) != expected_nullable
            ):
                return False
        primary_key = inspector.get_pk_constraint(table_name).get("constrained_columns")
        if tuple(primary_key or ()) != _LEGACY_V2_PRIMARY_KEYS[table_name]:
            return False
    rows = connection.execute(text("SELECT id, schema_version FROM schema_metadata")).all()
    return len(rows) == 1 and rows[0][0] == 1 and rows[0][1] == 2

###############################################################################
def _unsupported_schema_error(connection: Connection) -> RuntimeError:
    tables = sorted(set(inspect(connection).get_table_names()) - {ALEMBIC_VERSION_TABLE})
    found = ", ".join(tables) if tables else "none"
    return RuntimeError(
        "The database contains an unsupported or incomplete schema without a "
        "recognized Alembic revision. Supported unversioned adoption requires "
        "the normalized version-2 schema and its schema_metadata marker. "
        f"Tables found: {found}. Back up/export the data before recreating the database."
    )

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
def _validate_known_revisions(
    current_heads: tuple[str, ...], script: ScriptDirectory
) -> None:
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
                "Restore the matching migration files before starting the application."
            ) from exc

###############################################################################
def _run_migrations(connection: Connection) -> None:
    config = _alembic_config(connection)
    script, expected_head = _script_directory(config)
    current_heads = _current_heads(connection)
    _validate_known_revisions(current_heads, script)

    all_tables = set(inspect(connection).get_table_names())
    version_table_present = ALEMBIC_VERSION_TABLE in all_tables
    tables_without_version = all_tables - {ALEMBIC_VERSION_TABLE}
    if not current_heads and version_table_present:
        raise RuntimeError(
            "The database contains an Alembic version table without a current "
            "revision. It may be an incomplete migration; restore the matching "
            "migration state or recover from a backup before retrying."
        )
    if not current_heads and tables_without_version:
        if _legacy_v2_schema_matches(connection):
            LOGGER.warning(
                "Adopting an existing normalized version-2 database without an "
                "Alembic version marker"
            )
            command.stamp(config, BASELINE_REVISION)
            current_heads = (BASELINE_REVISION,)
        else:
            raise _unsupported_schema_error(connection)

    if current_heads == (expected_head,):
        LOGGER.info("Database schema is already at Alembic head %s", expected_head)
    else:
        if current_heads:
            LOGGER.info(
                "Upgrading database schema from %s to Alembic head %s",
                ", ".join(current_heads),
                expected_head,
            )
        else:
            LOGGER.info("Initializing an empty database to Alembic head %s", expected_head)
        command.upgrade(config, "head")

    final_heads = _current_heads(connection)
    if final_heads != (expected_head,):
        raise RuntimeError(
            "Database migration completed without reaching the expected Alembic head "
            f"{expected_head}; current heads are {final_heads or 'none'}."
        )
    actual_tables = frozenset(inspect(connection).get_table_names()) - {
        ALEMBIC_VERSION_TABLE
    }
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
    """Create or upgrade the configured SQLite database to the Alembic head."""
    backend = SQLiteBackend(settings)
    try:
        with (
            database_migration_lock(backend.db_path, settings.connect_timeout),
            backend.engine.begin() as connection,
        ):
            _run_migrations(connection)
    except Exception:
        LOGGER.exception(
            "Database initialization or migration failed; transaction rolled back"
        )
        backend.engine.dispose()
        raise
    return backend
