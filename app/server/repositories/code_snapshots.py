from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.enums import TargetFramework
from repositories.models import GeneratedCodeSnapshotRecord


class CodeSnapshotRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_revision(self, project_id: UUID, revision: int) -> dict[str, str]:
        rows = self._session.scalars(
            select(GeneratedCodeSnapshotRecord).where(
                GeneratedCodeSnapshotRecord.project_id == str(project_id),
                GeneratedCodeSnapshotRecord.revision == revision,
            )
        ).all()
        return {row.target_framework: row.source_code for row in rows}

    def upsert(
        self,
        project_id: UUID,
        *,
        revision: int,
        target_framework: TargetFramework,
        source_code: str,
        timestamp: datetime,
    ) -> None:
        key = {
            "project_id": str(project_id),
            "revision": revision,
            "target_framework": target_framework.value,
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
