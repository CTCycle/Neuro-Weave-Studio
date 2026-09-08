from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from repositories.models import ProjectRevisionRecord


###############################################################################
class RevisionRepository:
    # -------------------------------------------------------------------------
    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    def append(
        self,
        project_id: UUID,
        *,
        revision: int,
        graph: ModelGraph,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        timestamp: datetime,
    ) -> None:
        self._session.add(
            ProjectRevisionRecord(
                project_id=str(project_id),
                revision=revision,
                graph_schema_version=graph.schema_version,
                execution_backend=execution_backend.value,
                keras_enabled=keras_enabled,
                created_at=timestamp,
                graph_snapshot_json=graph.model_dump(mode="json"),
            )
        )
        self._session.flush()

    # -------------------------------------------------------------------------
    def get(self, project_id: UUID, revision: int) -> ProjectRevisionRecord | None:
        return self._session.scalar(
            select(ProjectRevisionRecord).where(
                ProjectRevisionRecord.project_id == str(project_id),
                ProjectRevisionRecord.revision == revision,
            )
        )
