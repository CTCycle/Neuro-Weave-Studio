from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from domain.enums import ExecutionBackend, ValidationStatus
from repositories.contracts import ProjectMetadata, RepositoryRevisionConflictError
from repositories.models import ProjectRecord

###############################################################################
class ProjectRepository:

    # -------------------------------------------------------------------------
    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    def list_metadata(self) -> list[ProjectMetadata]:
        records = self._session.scalars(
            select(ProjectRecord).order_by(ProjectRecord.updated_at.desc())
        ).all()
        return [self._to_metadata(record) for record in records]

    # -------------------------------------------------------------------------
    def get_metadata(self, project_id: UUID) -> ProjectMetadata | None:
        record = self._session.get(ProjectRecord, str(project_id))
        return self._to_metadata(record) if record is not None else None

    # -------------------------------------------------------------------------
    def create(
        self,
        *,
        name: str,
        description: str,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        graph_schema_version: int,
        timestamp: datetime,
        validation_status: ValidationStatus,
    ) -> ProjectMetadata:
        record = ProjectRecord(
            name=name,
            description=description,
            graph_schema_version=graph_schema_version,
            execution_backend=execution_backend.value,
            keras_enabled=keras_enabled,
            current_revision=1,
            validation_status=validation_status,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._session.add(record)
        self._session.flush()
        return self._to_metadata(record)

    # -------------------------------------------------------------------------
    def update(
        self,
        project_id: UUID,
        *,
        name: str,
        description: str,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        graph_schema_version: int,
        timestamp: datetime,
        validation_status: ValidationStatus,
        expected_revision: int,
    ) -> ProjectMetadata | None:
        result = self._session.execute(
            update(ProjectRecord)
            .where(
                ProjectRecord.id == str(project_id),
                ProjectRecord.current_revision == expected_revision,
            )
            .values(
                name=name,
                description=description,
                graph_schema_version=graph_schema_version,
                execution_backend=execution_backend.value,
                keras_enabled=keras_enabled,
                current_revision=expected_revision + 1,
                validation_status=validation_status,
                updated_at=timestamp,
            )
        )
        rowcount = getattr(result, "rowcount", -1)
        if rowcount == 0:
            record = self._session.get(ProjectRecord, str(project_id))
            if record is None:
                return None
            raise RepositoryRevisionConflictError(expected_revision, record.current_revision)
        self._session.flush()
        record = self._session.get(ProjectRecord, str(project_id))
        assert record is not None
        return self._to_metadata(record)

    # -------------------------------------------------------------------------
    def delete(self, project_id: UUID) -> bool:
        record = self._session.get(ProjectRecord, str(project_id))
        if record is None:
            return False
        self._session.delete(record)
        self._session.flush()
        return True

    # -------------------------------------------------------------------------
    @staticmethod
    def _to_metadata(record: ProjectRecord) -> ProjectMetadata:
        return ProjectMetadata(
            id=UUID(record.id),
            name=record.name,
            description=record.description,
            created_at=record.created_at,
            updated_at=record.updated_at,
            graph_schema_version=record.graph_schema_version,
            execution_backend=ExecutionBackend(record.execution_backend),
            keras_enabled=record.keras_enabled,
            current_revision=record.current_revision,
            validation_status=ValidationStatus(record.validation_status),
        )
