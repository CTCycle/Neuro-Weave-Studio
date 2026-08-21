from __future__ import annotations

import re
from uuid import UUID

from common.catalogs.loader import NodeCatalog
from configurations.settings import Settings
from domain.diagnostics import ValidationDiagnostic
from domain.enums import DiagnosticSeverity
from domain.graph import ModelGraph
from domain.jobs import JobResponse, JobSubmitRequest, SmokeInferenceOperation
from domain.runtime import WorkerConfiguration
from services.errors import CompilationRejectedError, ProjectNotFoundError
from services.project_service import ProjectService
from services.validation_service import ValidationService
from models.runtime_controller import RuntimeController

###############################################################################
class JobService:
    """Validates graph/project inputs before handing immutable work to the controller."""

    def __init__(
        self,
        catalog: NodeCatalog,
        projects: ProjectService,
        controller: RuntimeController,
        settings: Settings,
    ) -> None:
        self._validator = ValidationService(catalog)
        self._projects = projects
        self._controller = controller
        self._settings = settings

    # -------------------------------------------------------------------------
    def submit(self, request: JobSubmitRequest) -> JobResponse:
        if isinstance(request.operation, SmokeInferenceOperation):
            graph = request.operation.graph
            validation = self._validator.validate(graph, request.worker.backend, True)
            if not validation.valid:
                raise CompilationRejectedError(validation.diagnostics)
            worker = self._with_graph_precision(request.worker, graph)
            return self._controller.submit(
                request.model_copy(update={"worker": worker}),
                graph=graph,
            )

        project_id = request.operation.project_id
        project = self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        if not project.keras_enabled:
            raise CompilationRejectedError(
                [ValidationDiagnostic(
                    code="native_runtime_unavailable",
                    severity=DiagnosticSeverity.COMPATIBILITY,
                    message="Native runtime execution is not implemented",
                )]
            )
        validation = self._validator.validate(
            project.graph, project.execution_backend, project.keras_enabled
        )
        if not validation.valid:
            raise CompilationRejectedError(validation.diagnostics)
        worker = self._with_graph_precision(request.worker, project.graph)
        destination = (
            self._settings.export_root
            / str(project_id)
            / f"{self._safe_filename_stem(project.name)}.keras"
        )
        return self._controller.submit(
            request.model_copy(update={"worker": worker}),
            graph=project.graph,
            artifact_destination=destination,
            artifact_filename=destination.name,
            project_revision=project.current_revision,
            revision_checker=lambda: self._project_revision_is_current(
                project_id, project.current_revision
            ),
        )

    # -------------------------------------------------------------------------
    def _project_revision_is_current(self, project_id: UUID, revision: int) -> bool:
        project = self._projects.get(project_id)
        return project is not None and project.current_revision == revision

    # -------------------------------------------------------------------------
    @staticmethod
    def _with_graph_precision(
        worker: WorkerConfiguration, graph: ModelGraph
    ) -> WorkerConfiguration:
        if worker.precision != "float32":
            return worker
        if graph.model_configuration.mixed_precision:
            return worker.model_copy(update={"precision": "mixed_float16"})
        return worker

    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_filename_stem(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return (sanitized or "model")[:120]
