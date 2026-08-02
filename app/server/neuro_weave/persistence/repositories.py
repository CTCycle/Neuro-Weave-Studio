from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from neuro_weave.domain.enums import TargetFramework
from neuro_weave.domain.graph import ModelGraph, ProjectDocument, ProjectSummary
from neuro_weave.persistence.models import ProjectRecord, ProjectRevisionRecord
from neuro_weave.persistence.session import RepositoryContext


class ProjectRepository:
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context

    @property
    def _session_factory(self):
        return self.context.session_factory

    @contextmanager
    def _write_session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list(self) -> list[ProjectSummary]:
        with self._session_factory() as session:
            records = session.scalars(
                select(ProjectRecord).order_by(ProjectRecord.updated_at.desc())
            ).all()
            return [self._to_summary(record) for record in records]

    def get(self, project_id: UUID) -> ProjectDocument | None:
        with self._session_factory() as session:
            record = session.get(ProjectRecord, str(project_id))
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
        with self._write_session() as session:
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
            session.add(record)
            session.flush()
            session.refresh(record)
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
        graph_json = graph.model_dump_json()
        with self._write_session() as session:
            if expected_revision is None:
                record = session.get(ProjectRecord, str(project_id))
            else:
                record = session.scalar(
                    select(ProjectRecord)
                    .where(ProjectRecord.id == str(project_id))
                    .with_for_update()
                )
            if record is None:
                return None
            if expected_revision is not None and record.current_revision != expected_revision:
                raise ValueError(
                    f"Revision conflict: expected {expected_revision}, current {record.current_revision}"
                )
            revision = record.current_revision + 1
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
            session.flush()
            session.refresh(record)
            return self._to_document(record)

    def delete(self, project_id: UUID) -> bool:
        with self._write_session() as session:
            record = session.get(ProjectRecord, str(project_id))
            if record is None:
                return False
            session.delete(record)
            return True

    def save_code_snapshot(self, project_id: UUID, framework: TargetFramework, code: str) -> None:
        with self._write_session() as session:
            record = session.get(ProjectRecord, str(project_id))
            if record is None:
                raise KeyError(str(project_id))
            snapshots = json.loads(record.generated_code_json)
            snapshots[framework.value] = code
            record.generated_code_json = json.dumps(snapshots, sort_keys=True)

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
