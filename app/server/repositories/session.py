# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from configurations.settings import DatabaseSettings
from repositories.backend import SQLiteBackend, build_session_factory
from repositories.initializer import initialize_database
from repositories.unit_of_work import UnitOfWork


###############################################################################
@dataclass(frozen=True, slots=True)
class RepositoryContext:
    """Immutable database handles shared by the application composition root."""

    backend: SQLiteBackend
    engine: Engine
    session_factory: sessionmaker[Session]

    # -------------------------------------------------------------------------
    def new_unit_of_work(self) -> UnitOfWork:
        return UnitOfWork(self.session_factory)

    # -------------------------------------------------------------------------
    def dispose(self) -> None:
        self.engine.dispose()

###############################################################################
def create_repository_context(settings: DatabaseSettings) -> RepositoryContext:
    backend = initialize_database(settings)
    return RepositoryContext(
        backend=backend,
        engine=backend.engine,
        session_factory=build_session_factory(backend.engine),
    )
