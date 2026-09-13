from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from domain.enums import ExecutionBackend
from repositories.models import GeneratedCodeSnapshotRecord

###############################################################################
class CodeSnapshotRepository:

    # -------------------------------------------------------------------------
    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    def get_for_revision(self, project_id: UUID, revision: int) -> str | None:
        row = self._session.scalar(
            select(GeneratedCodeSnapshotRecord).where(
                GeneratedCodeSnapshotRecord.project_id == str(project_id),
                GeneratedCodeSnapshotRecord.revision == revision,
            )
        )
        return row.source_code if row is not None else None

    # -------------------------------------------------------------------------
    def has_for_revisions(self, revisions: Mapping[UUID, int]) -> dict[UUID, bool]:
        if not revisions:
            return {}
        conditions = [
            and_(
                GeneratedCodeSnapshotRecord.project_id == str(project_id),
                GeneratedCodeSnapshotRecord.revision == revision,
            )
            for project_id, revision in revisions.items()
        ]
        rows = self._session.scalars(
            select(GeneratedCodeSnapshotRecord).where(or_(*conditions))
        ).all()
        result: dict[UUID, bool] = {}
        for row in rows:
            project_id = UUID(row.project_id)
            result[project_id] = True
        return result

    # -------------------------------------------------------------------------
    def upsert(
        self,
        project_id: UUID,
        *,
        revision: int,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        source_code: str,
        timestamp: datetime,
    ) -> None:
        key = {
            "project_id": str(project_id),
            "revision": revision,
        }
        row = self._session.get(GeneratedCodeSnapshotRecord, key)
        if row is None:
            self._session.add(
                GeneratedCodeSnapshotRecord(
                    **key,
                    execution_backend=execution_backend.value,
                    keras_enabled=keras_enabled,
                    source_code=source_code,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
        else:
            row.source_code = source_code
            row.updated_at = timestamp
        self._session.flush()

    # -------------------------------------------------------------------------
    def delete_for_projects(self, project_ids: Sequence[UUID]) -> int:
        if not project_ids:
            return 0
        result = cast(
            CursorResult[Any],
            self._session.execute(
                delete(GeneratedCodeSnapshotRecord).where(
                    GeneratedCodeSnapshotRecord.project_id.in_([str(item) for item in project_ids])
                )
            ),
        )
        return int(result.rowcount or 0)
