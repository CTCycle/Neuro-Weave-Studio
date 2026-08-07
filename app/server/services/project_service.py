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


###############################################################################
class ProjectService:

    # -------------------------------------------------------------------------
    def __init__(self, uow_factory: UnitOfWorkFactory, validator: ValidationService) -> None:
        self._uow_factory = uow_factory
        self._validator = validator

    # -------------------------------------------------------------------------
    def list(self) -> ProjectListResponse:
        with self._uow_factory() as uow:
            metadata = uow.projects.list_metadata()
            graph_summaries = uow.graphs.summaries([item.id for item in metadata])
            compiled_targets = uow.code_snapshots.targets_for_revisions(
                {item.id: item.current_revision for item in metadata}
            )
            summaries = [
                ProjectSummary(
                    id=item.id,
                    name=item.name,
                    description=item.description,
                    updated_at=item.updated_at,
                    code_target=item.code_target,
                    current_revision=item.current_revision,
                    validation_status=item.validation_status,
                    node_count=graph_summaries[item.id].node_count,
                    edge_count=graph_summaries[item.id].edge_count,
                    compiled_targets=compiled_targets.get(item.id, []),
                )
                for item in metadata
            ]
        return ProjectListResponse(projects=summaries)

    # -------------------------------------------------------------------------
    def get(self, project_id: UUID) -> ProjectDocument | None:
        with self._uow_factory() as uow:
            metadata = uow.projects.get_metadata(project_id)
            if metadata is None:
                return None
            return self._document(uow, metadata)

    # -------------------------------------------------------------------------
    def create(self, payload: ProjectCreate) -> ProjectDocument:
        validation = self._validator.validate(payload.graph, payload.code_target)
        timestamp = utc_now()
        try:
            with self._uow_factory() as uow:
                metadata = uow.projects.create(
                    name=payload.name,
                    description=payload.description,
                    code_target=payload.code_target,
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
                    code_target=payload.code_target,
                    timestamp=timestamp,
                )
                document = self._document(uow, metadata)
                uow.commit()
                return document
        except IntegrityError as exc:
            raise GraphPersistenceError(
                "Project graph conflicts with the relational schema"
            ) from exc

    # -------------------------------------------------------------------------
    def update(self, project_id: UUID, payload: ProjectUpdate) -> ProjectDocument | None:
        validation = self._validator.validate(payload.graph, payload.code_target)
        timestamp = utc_now()
        try:
            with self._uow_factory() as uow:
                metadata = uow.projects.update(
                    project_id,
                    name=payload.name,
                    description=payload.description,
                    code_target=payload.code_target,
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
                    code_target=payload.code_target,
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

    # -------------------------------------------------------------------------
    def delete(self, project_id: UUID) -> bool:
        with self._uow_factory() as uow:
            deleted = uow.projects.delete(project_id)
            if deleted:
                uow.commit()
            return deleted

    # -------------------------------------------------------------------------
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
                code_target=source.code_target,
                graph=source.graph,
            )
        )

    # -------------------------------------------------------------------------
    def export_bundle(self, project_id: UUID) -> ProjectExportBundle | None:
        project = self.get(project_id)
        if project is None:
            return None
        return ProjectExportBundle(
            project=ProjectCreate(
                name=project.name,
                description=project.description,
                code_target=project.code_target,
                graph=project.graph,
            )
        )

    # -------------------------------------------------------------------------
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
            code_target=metadata.code_target,
            current_revision=metadata.current_revision,
            graph=graph,
            validation_status=metadata.validation_status,
            generated_code_snapshots=uow.code_snapshots.list_for_revision(
                metadata.id, metadata.current_revision
            ),
        )
