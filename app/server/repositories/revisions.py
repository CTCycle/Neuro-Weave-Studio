# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

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
        timestamp: datetime,
    ) -> None:
        self._session.add(
            ProjectRevisionRecord(
                project_id=str(project_id),
                revision=revision,
                created_at=timestamp,
            )
        )
        self._session.flush()
