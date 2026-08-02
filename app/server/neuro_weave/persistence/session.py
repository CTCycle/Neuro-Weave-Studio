from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from neuro_weave.config import get_database_settings
from neuro_weave.persistence.backend import DatabaseBackend, build_session_factory
from neuro_weave.persistence.initializer import ensure_database_ready


@dataclass(frozen=True, slots=True)
class RepositoryContext:
    """Immutable database handles shared by the application composition root."""

    backend: DatabaseBackend
    engine: Engine
    session_factory: sessionmaker[Session]

    def dispose(self) -> None:
        self.engine.dispose()


@lru_cache(maxsize=1)
def get_repository_context() -> RepositoryContext:
    backend = ensure_database_ready(get_database_settings())
    return RepositoryContext(
        backend=backend,
        engine=backend.engine,
        session_factory=build_session_factory(backend.engine),
    )


def close_repository_context() -> None:
    if get_repository_context.cache_info().currsize:
        get_repository_context().dispose()
    get_repository_context.cache_clear()
