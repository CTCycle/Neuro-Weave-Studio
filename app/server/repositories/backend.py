from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from configurations.settings import DatabaseSettings

POSTGRES_DATABASE_NAME_RE = re.compile(r"^\w[\w-]{0,62}$", re.ASCII)

###############################################################################
def normalize_postgres_engine(engine: str | None) -> str:
    normalized = (engine or "postgresql+psycopg").casefold()
    if normalized in {"postgres", "postgresql"}:
        return "postgresql+psycopg"
    return normalized

###############################################################################
def validate_postgres_database_name(database_name: str) -> str:
    normalized = str(database_name or "").strip()
    if not POSTGRES_DATABASE_NAME_RE.fullmatch(normalized):
        raise ValueError("Invalid PostgreSQL database name")
    return normalized

###############################################################################
def build_postgres_url(settings: DatabaseSettings, database_name: str | None = None) -> str:
    if not settings.host or not settings.username:
        raise ValueError("PostgreSQL host and username are required")
    target_name = validate_postgres_database_name(
        database_name or settings.database_name or ""
    )
    return (
        f"{normalize_postgres_engine(settings.engine)}://"
        f"{quote_plus(settings.username)}:{quote_plus(settings.password or '')}"
        f"@{settings.host}:{settings.port or 5432}/{target_name}"
    )

###############################################################################
def build_postgres_connect_args(settings: DatabaseSettings) -> dict[str, Any]:
    connect_args: dict[str, Any] = {
        "connect_timeout": settings.connect_timeout,
        "client_encoding": "utf8",
    }
    if settings.ssl:
        connect_args["sslmode"] = "require"
        if settings.ssl_ca:
            connect_args["sslrootcert"] = settings.ssl_ca
    return connect_args

###############################################################################
class SQLiteBackend:

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        if settings.sqlite_path is None:
            raise ValueError("SQLite path is required for embedded database mode")
        self.db_path = Path(settings.sqlite_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite+pysqlite:///{self.db_path.as_posix()}",
            echo=False,
            future=True,
            connect_args={
                "check_same_thread": False,
                "timeout": settings.connect_timeout,
            },
        )
        event.listen(self.engine, "connect", self._configure_connection)

    # -------------------------------------------------------------------------
    @staticmethod
    def _configure_connection(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()

###############################################################################
class PostgresBackend:

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        if not settings.host:
            raise ValueError("Database host must be provided for external database")
        if not settings.database_name:
            raise ValueError("Database name must be provided for external database")
        if not settings.username:
            raise ValueError("Database username must be provided for external database")
        self.db_path: None = None
        self.engine = create_engine(
            build_postgres_url(settings),
            echo=False,
            future=True,
            connect_args=build_postgres_connect_args(settings),
            pool_pre_ping=True,
            pool_timeout=30,
            pool_recycle=1800,
        )


DatabaseBackend = SQLiteBackend | PostgresBackend

###############################################################################
def build_database_backend(settings: DatabaseSettings) -> DatabaseBackend:
    if settings.embedded_database:
        return SQLiteBackend(settings)
    if normalize_postgres_engine(settings.engine) != "postgresql+psycopg":
        raise ValueError(f"Unsupported database engine: {settings.engine}")
    return PostgresBackend(settings)

###############################################################################
def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
