from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from common.catalogs.loader import get_node_catalog
from common.timestamps import utc_now
from domain.enums import ExecutionBackend, ValidationStatus
from domain.graph import (
    ProjectCreate,
    ProjectDocument,
    ProjectDuplicateRequest,
    ProjectExportBundle,
    ProjectListResponse,
    ProjectSummary,
    ProjectUpdate,
)
from domain.metadata import ExecutionBackendStatus
from repositories.contracts import (
    ProjectMetadata,
    RepositoryIntegrityError,
    RepositoryRevisionConflictError,
    UnitOfWorkFactory,
    UnitOfWorkProtocol,
)
from services.errors import (
    GraphPersistenceError,
    RevisionConflictError,
)
from services.validation_service import ValidationService
from validation.output_resolution import input_nodes, is_effective_output, resolved_outputs


###############################################################################
class ProjectService:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        validator: ValidationService,
        backend_statuses: Mapping[ExecutionBackend, ExecutionBackendStatus] | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._validator = validator
        self._backend_statuses = backend_statuses or {}

    # -------------------------------------------------------------------------
    def list(self) -> ProjectListResponse:
        catalog = get_node_catalog()
        with self._uow_factory() as uow:
            metadata = uow.projects.list_metadata()
            graphs = uow.graphs.load_many(
                {item.id: item.graph_schema_version for item in metadata}
            )
            generated_sources = uow.code_snapshots.has_for_revisions(
                {item.id: item.current_revision for item in metadata}
            )
            summaries: list[ProjectSummary] = []
            for item in metadata:
                graph = graphs.get(item.id)
                if graph is None:
                    raise RuntimeError(f"Project {item.id} is missing graph settings")
                summaries.append(
                    ProjectSummary(
                        id=item.id,
                        name=item.name,
                        description=item.description,
                        updated_at=item.updated_at,
                        execution_backend=item.execution_backend,
                        keras_enabled=item.keras_enabled,
                        current_revision=item.current_revision,
                        validation_status=item.validation_status,
                        input_count=len(input_nodes(graph, catalog)),
                        output_count=sum(
                            is_effective_output(output)
                            for output in resolved_outputs(graph, catalog)
                        ),
                        node_count=len(graph.nodes),
                        edge_count=len(graph.edges),
                        has_generated_source=generated_sources.get(item.id, False),
                    )
                )
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
        execution_backend = self._resolve_execution_backend(payload.execution_backend)
        validation = self._validator.validate(
            payload.graph, execution_backend, payload.keras_enabled
        )
        timestamp = utc_now()
        try:
            with self._uow_factory() as uow:
                metadata = uow.projects.create(
                    name=payload.name,
                    description=payload.description,
                    execution_backend=execution_backend,
                    keras_enabled=payload.keras_enabled,
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
                    execution_backend=execution_backend,
                    keras_enabled=payload.keras_enabled,
                    timestamp=timestamp,
                )
                document = self._document(uow, metadata)
                uow.commit()
                return document
        except RepositoryIntegrityError as exc:
            raise GraphPersistenceError(
                "Project graph conflicts with the relational schema"
            ) from exc

    # -------------------------------------------------------------------------
    def update(self, project_id: UUID, payload: ProjectUpdate) -> ProjectDocument | None:
        validation = self._validator.validate(
            payload.graph, payload.execution_backend, payload.keras_enabled
        )
        timestamp = utc_now()
        try:
            with self._uow_factory() as uow:
                metadata = uow.projects.update(
                    project_id,
                    name=payload.name,
                    description=payload.description,
                    execution_backend=payload.execution_backend,
                    keras_enabled=payload.keras_enabled,
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
                    execution_backend=payload.execution_backend,
                    keras_enabled=payload.keras_enabled,
                    timestamp=timestamp,
                )
                document = self._document(uow, metadata)
                uow.commit()
                return document
        except RepositoryIntegrityError as exc:
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
                execution_backend=source.execution_backend,
                keras_enabled=source.keras_enabled,
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
                execution_backend=project.execution_backend,
                keras_enabled=project.keras_enabled,
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
            execution_backend=metadata.execution_backend,
            keras_enabled=metadata.keras_enabled,
            current_revision=metadata.current_revision,
            graph=graph,
            validation_status=metadata.validation_status,
            generated_source=uow.code_snapshots.get_for_revision(
                metadata.id, metadata.current_revision
            ),
        )

    # -------------------------------------------------------------------------
    def _resolve_execution_backend(self, requested: ExecutionBackend | None) -> ExecutionBackend:
        if requested is not None:
            return requested
        available = {
            backend for backend, status in self._backend_statuses.items() if status.available
        }
        if ExecutionBackend.PYTORCH in available:
            return ExecutionBackend.PYTORCH
        for backend in ExecutionBackend:
            if backend in available:
                return backend
        return ExecutionBackend.PYTORCH
