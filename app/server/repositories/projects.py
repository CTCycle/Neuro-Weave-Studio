from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.enums import TargetFramework
from domain.graph import ModelGraph, ProjectDocument, ProjectSummary
from repositories.models import ProjectRecord, ProjectRevisionRecord


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list(self) -> list[ProjectSummary]:
        records = self._session.scalars(select(ProjectRecord).order_by(ProjectRecord.updated_at.desc())).all()
        return [self._to_summary(record) for record in records]

    def get(self, project_id: UUID) -> ProjectDocument | None:
        record = self._session.get(ProjectRecord, str(project_id))
        return self._to_document(record) if record is not None else None

    def create(
        self,
        *,
        name: str,
        description: str,
        target_framework: TargetFramework,
        graph: ModelGraph,
        timestamp: datetime,
        validation_status: str,
    ) -> ProjectDocument:
        graph_json = graph.model_dump_json()
        record = ProjectRecord(
            name=name,
            description=description,
            created_at=timestamp,
            updated_at=timestamp,
            schema_version=graph.schema_version,
            target_framework=target_framework.value,
            current_revision=1,
            validation_status=validation_status,
            graph_json=graph_json,
            generated_code_json="{}",
        )
        record.revisions.append(
            ProjectRevisionRecord(
                revision=1,
                created_at=timestamp,
                graph_json=graph_json,
                target_framework=target_framework.value,
            )
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return self._to_document(record)

    def update(
        self,
        project_id: UUID,
        *,
        name: str,
        description: str,
        target_framework: TargetFramework,
        graph: ModelGraph,
        timestamp: datetime,
        validation_status: str,
        expected_revision: int | None,
    ) -> ProjectDocument | None:
        record = self._session.get(ProjectRecord, str(project_id))
        if record is None:
            return None
        if expected_revision is not None and record.current_revision != expected_revision:
            raise ValueError(
                f"Revision conflict: expected {expected_revision}, current {record.current_revision}"
            )
        revision = record.current_revision + 1
        graph_json = graph.model_dump_json()
        record.name = name
        record.description = description
        record.target_framework = target_framework.value
        record.graph_json = graph_json
        record.schema_version = graph.schema_version
        record.current_revision = revision
        record.updated_at = timestamp
        record.validation_status = validation_status
        record.revisions.append(
            ProjectRevisionRecord(
                revision=revision,
                created_at=timestamp,
                graph_json=graph_json,
                target_framework=target_framework.value,
            )
        )
        self._session.commit()
        self._session.refresh(record)
        return self._to_document(record)

    def delete(self, project_id: UUID) -> bool:
        record = self._session.get(ProjectRecord, str(project_id))
        if record is None:
            return False
        self._session.delete(record)
        self._session.commit()
        return True

    def save_code_snapshot(self, project_id: UUID, framework: TargetFramework, code: str) -> None:
        record = self._session.get(ProjectRecord, str(project_id))
        if record is None:
            raise KeyError(str(project_id))
        snapshots = json.loads(record.generated_code_json)
        snapshots[framework.value] = code
        record.generated_code_json = json.dumps(snapshots, sort_keys=True)
        self._session.commit()

    def _to_document(self, record: ProjectRecord) -> ProjectDocument:
        return ProjectDocument(
            id=UUID(record.id),
            name=record.name,
            description=record.description,
            created_at=record.created_at,
            updated_at=record.updated_at,
            schema_version=record.schema_version,
            target_framework=TargetFramework(record.target_framework),
            current_revision=record.current_revision,
            graph=ModelGraph.model_validate_json(record.graph_json),
            validation_status=record.validation_status,
            generated_code_snapshots=json.loads(record.generated_code_json),
        )

    @staticmethod
    def _to_summary(record: ProjectRecord) -> ProjectSummary:
        return ProjectSummary(
            id=UUID(record.id),
            name=record.name,
            description=record.description,
            updated_at=record.updated_at,
            target_framework=TargetFramework(record.target_framework),
            current_revision=record.current_revision,
            validation_status=record.validation_status,
        )
