# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

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
        result: dict[UUID, bool] = {}
        items = list(revisions.items())
        for start in range(0, len(items), 250):
            conditions = [
                and_(
                    GeneratedCodeSnapshotRecord.project_id == str(project_id),
                    GeneratedCodeSnapshotRecord.revision == revision,
                )
                for project_id, revision in items[start : start + 250]
            ]
            project_ids = self._session.scalars(
                select(GeneratedCodeSnapshotRecord.project_id).where(or_(*conditions))
            )
            result.update({UUID(project_id): True for project_id in project_ids})
        return result

    # -------------------------------------------------------------------------
    def upsert(
        self,
        project_id: UUID,
        *,
        revision: int,
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
