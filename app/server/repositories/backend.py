from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from configurations.settings import DatabaseSettings

###############################################################################
class SQLiteBackend:

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path = Path(settings.sqlite_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = settings.connect_timeout * 1000
        self.engine = create_engine(
            build_sqlite_url(self.db_path),
            echo=False,
            future=True,
            connect_args={
                "check_same_thread": False,
                "timeout": settings.connect_timeout,
                "autocommit": False,
            },
        )
        event.listen(self.engine, "connect", self._configure_connection)

    # -------------------------------------------------------------------------
    def _configure_connection(self, dbapi_connection: Any, _connection_record: Any) -> None:
        previous_autocommit = getattr(dbapi_connection, "autocommit", None)
        if previous_autocommit is not None:
            dbapi_connection.autocommit = True
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        finally:
            cursor.close()
            if previous_autocommit is not None:
                dbapi_connection.autocommit = previous_autocommit

###############################################################################
def build_sqlite_url(db_path: Path) -> str:
    """Build the SQLite URL shared by the application and Alembic."""
    return f"sqlite+pysqlite:///{db_path.resolve().as_posix()}"

###############################################################################
def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
