from __future__ import annotations

from types import TracebackType
from typing import Self

from sqlalchemy.orm import Session, sessionmaker

from repositories.code_snapshots import CodeSnapshotRepository
from repositories.graphs import GraphRepository
from repositories.projects import ProjectRepository
from repositories.revisions import RevisionRepository


class UnitOfWork:
    """One explicit session boundary shared by all repositories in an operation."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.session: Session | None = None
        self._committed = False

    def __enter__(self) -> Self:
        self.session = self._session_factory()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.session is None:
            return
        if exc_type is not None or not self._committed:
            self.session.rollback()
        self.session.close()

    @property
    def projects(self) -> ProjectRepository:
        return ProjectRepository(self._require_session())

    @property
    def graphs(self) -> GraphRepository:
        return GraphRepository(self._require_session())

    @property
    def revisions(self) -> RevisionRepository:
        return RevisionRepository(self._require_session())

    @property
    def code_snapshots(self) -> CodeSnapshotRepository:
        return CodeSnapshotRepository(self._require_session())

    def commit(self) -> None:
        self._require_session().commit()
        self._committed = True

    def _require_session(self) -> Session:
        if self.session is None:
            raise RuntimeError("Unit of work must be entered before repositories are used")
        return self.session
