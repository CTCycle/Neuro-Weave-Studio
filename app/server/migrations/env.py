# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from logging.config import fileConfig
from typing import cast

from alembic import context
from sqlalchemy.engine import Connection

from configurations.settings import get_database_settings
from repositories import (
    custom_nodes as _custom_nodes,  # pyright: ignore[reportUnusedImport]  # noqa: F401 - register ORM tables
)
from repositories import (
    models as _models,  # pyright: ignore[reportUnusedImport]  # noqa: F401 - register ORM tables
)
from repositories.backend import SQLiteBackend, build_sqlite_url
from repositories.base import Base
from repositories.migration_lock import database_migration_lock

config = context.config

if config.config_file_name is not None and not config.attributes.get("skip_logging_config", False):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

###############################################################################
def _configure_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=True,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()

###############################################################################
def run_migrations_offline() -> None:
    settings = get_database_settings()
    context.configure(
        url=build_sqlite_url(settings.sqlite_path),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()

###############################################################################
def run_migrations_online() -> None:
    supplied_connection = cast(Connection | None, config.attributes.get("connection"))
    if supplied_connection is not None:
        _configure_migrations(supplied_connection)
        return

    settings = get_database_settings()
    backend = SQLiteBackend(settings)
    try:
        with (
            database_migration_lock(backend.db_path, settings.connect_timeout),
            backend.engine.begin() as connection,
        ):
            _configure_migrations(connection)
    finally:
        backend.engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
