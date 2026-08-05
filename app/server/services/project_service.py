from __future__ import annotations

from uuid import UUID

from sqlalchemy.exc import IntegrityError

from common.timestamps import utc_now
from domain.enums import ValidationStatus
from domain.graph import (
    ProjectCreate,
    ProjectDocument,
    ProjectDuplicateRequest,
    ProjectExportBundle,
    ProjectListResponse,
    ProjectSummary,
    ProjectUpdate,
)
from repositories.contracts import (
    ProjectMetadata,
    RepositoryRevisionConflictError,
    UnitOfWorkFactory,
    UnitOfWorkProtocol,
)
from services.errors import (
    GraphPersistenceError,
    RevisionConflictError,
)
from services.validation_service import ValidationService


class ProjectService:
    def __init__(self, uow_factory: UnitOfWorkFactory, validator: ValidationService) -> None:
        self._uow_factory = uow_factory
        self._validator = validator

    def list(self) -> ProjectListResponse:
        with self._uow_factory() as uow:
            summaries = [
                ProjectSummary(
                    id=item.id,
                    name=item.name,
                    description=item.description,
                    updated_at=item.updated_at,
                    target_framework=item.target_framework,
                    current_revision=item.current_revision,
                    validation_status=item.validation_status,
                )
                for item in uow.projects.list_metadata()
            ]
        return ProjectListResponse(projects=summaries)

    def get(self, project_id: UUID) -> ProjectDocument | None:
        with self._uow_factory() as uow:
            metadata = uow.projects.get_metadata(project_id)
            if metadata is None:
                return None
            return self._document(uow, metadata)

    def create(self, payload: ProjectCreate) -> ProjectDocument:
        validation = self._validator.validate(payload.graph, payload.target_framework)
        timestamp = utc_now()
        try:
            with self._uow_factory() as uow:
                metadata = uow.projects.create(
                    name=payload.name,
                    description=payload.description,
                    target_framework=payload.target_framework,
                    graph_schema_version=payload.graph.schema_version,
                    timestamp=timestamp,
                    validation_status=(
                        ValidationStatus.VALID if validation.valid else ValidationStatus.INVALID
                    ),
                )
                uow.graphs.replace(metadata.id, payload.graph)
                uow.revisions.append(
                    metadata.id,
                    revision=1,
                    graph=payload.graph,
                    target_framework=payload.target_framework,
                    timestamp=timestamp,
                )
                document = self._document(uow, metadata)
                uow.commit()
                return document
        except IntegrityError as exc:
            raise GraphPersistenceError(
                "Project graph conflicts with the relational schema"
            ) from exc

    def update(self, project_id: UUID, payload: ProjectUpdate) -> ProjectDocument | None:
        validation = self._validator.validate(payload.graph, payload.target_framework)
        timestamp = utc_now()
        try:
            with self._uow_factory() as uow:
                metadata = uow.projects.update(
                    project_id,
                    name=payload.name,
                    description=payload.description,
                    target_framework=payload.target_framework,
                    graph_schema_version=payload.graph.schema_version,
                    timestamp=timestamp,
                    validation_status=(
                        ValidationStatus.VALID if validation.valid else ValidationStatus.INVALID
                    ),
                    expected_revision=payload.expected_revision,
                )
                if metadata is None:
                    return None
                uow.graphs.replace(metadata.id, payload.graph)
                uow.revisions.append(
                    metadata.id,
                    revision=metadata.current_revision,
                    graph=payload.graph,
                    target_framework=payload.target_framework,
                    timestamp=timestamp,
                )
                document = self._document(uow, metadata)
                uow.commit()
                return document
        except IntegrityError as exc:
            raise GraphPersistenceError(
                "Project graph conflicts with the relational schema"
            ) from exc
        except RepositoryRevisionConflictError as exc:
            raise RevisionConflictError(str(exc)) from exc

    def delete(self, project_id: UUID) -> bool:
        with self._uow_factory() as uow:
            deleted = uow.projects.delete(project_id)
            if deleted:
                uow.commit()
            return deleted

    def duplicate(
        self, project_id: UUID, payload: ProjectDuplicateRequest
    ) -> ProjectDocument | None:
        source = self.get(project_id)
        if source is None:
            return None
        return self.create(
            ProjectCreate(
                name=payload.name or f"{source.name} copy",
                description=source.description,
                target_framework=source.target_framework,
                graph=source.graph,
            )
        )

    def export_bundle(self, project_id: UUID) -> ProjectExportBundle | None:
        project = self.get(project_id)
        if project is None:
            return None
        return ProjectExportBundle(
            project=ProjectCreate(
                name=project.name,
                description=project.description,
                target_framework=project.target_framework,
                graph=project.graph,
            )
        )

    @staticmethod
    def _document(uow: UnitOfWorkProtocol, metadata: ProjectMetadata) -> ProjectDocument:
        graph = uow.graphs.load(metadata.id, metadata.graph_schema_version)
        if graph is None:
            raise RuntimeError(f"Project {metadata.id} is missing graph settings")
        return ProjectDocument(
            id=metadata.id,
            name=metadata.name,
            description=metadata.description,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            schema_version=metadata.graph_schema_version,
            target_framework=metadata.target_framework,
            current_revision=metadata.current_revision,
            graph=graph,
            validation_status=metadata.validation_status,
            generated_code_snapshots=uow.code_snapshots.list_for_revision(
                metadata.id, metadata.current_revision
            ),
        )
