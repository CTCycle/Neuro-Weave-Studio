from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from domain.enums import CodeTarget
from repositories.models import GeneratedCodeSnapshotRecord


###############################################################################
class CodeSnapshotRepository:

    # -------------------------------------------------------------------------
    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    def list_for_revision(self, project_id: UUID, revision: int) -> dict[str, str]:
        rows = self._session.scalars(
            select(GeneratedCodeSnapshotRecord).where(
                GeneratedCodeSnapshotRecord.project_id == str(project_id),
                GeneratedCodeSnapshotRecord.revision == revision,
            )
        ).all()
        return {row.code_target: row.source_code for row in rows}

    # -------------------------------------------------------------------------
    def targets_for_revisions(
        self, revisions: Mapping[UUID, int]
    ) -> dict[UUID, list[CodeTarget]]:
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
        result: dict[UUID, list[CodeTarget]] = {}
        for row in rows:
            project_id = UUID(row.project_id)
            result.setdefault(project_id, []).append(CodeTarget(row.code_target))
        return result

    # -------------------------------------------------------------------------
    def upsert(
        self,
        project_id: UUID,
        *,
        revision: int,
        code_target: CodeTarget,
        source_code: str,
        timestamp: datetime,
    ) -> None:
        key = {
            "project_id": str(project_id),
            "revision": revision,
            "code_target": code_target.value,
        }
        row = self._session.get(GeneratedCodeSnapshotRecord, key)
        if row is None:
            self._session.add(
                GeneratedCodeSnapshotRecord(
                    **key,
                    source_code=source_code,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
        else:
            row.source_code = source_code
            row.updated_at = timestamp
        self._session.flush()
