from __future__ import annotations

import re
from uuid import UUID

from configurations.settings import Settings
from domain.compilation_catalog import CompilationCatalogProvider, get_compilation_catalog_provider
from domain.diagnostics import ValidationDiagnostic
from domain.enums import DiagnosticSeverity
from domain.exports import ExportFormat, ExportOptions, export_suffix
from domain.graph import ProjectDocument
from domain.jobs import JobResponse, JobSubmitRequest, ModelExportOperation, SmokeInferenceOperation
from domain.runtime import RuntimeApi, WorkerConfiguration
from services.errors import CompilationRejectedError, ProjectNotFoundError
from services.export_service import ExportService
from services.project_service import ProjectService
from services.runtime_controller import RuntimeController
from services.validation_service import ValidationService


###############################################################################
class JobService:
    """Validates graph/project inputs before handing immutable work to the controller."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        projects: ProjectService,
        controller: RuntimeController,
        settings: Settings,
        validator: ValidationService,
        exports: ExportService,
        compilation_catalog_provider: CompilationCatalogProvider | None = None,
    ) -> None:
        self._validator = validator
        self._projects = projects
        self._controller = controller
        self._settings = settings
        self._exports = exports
        self._compilation_catalog_provider = compilation_catalog_provider or get_compilation_catalog_provider()

    # -------------------------------------------------------------------------
    def submit(self, request: JobSubmitRequest) -> JobResponse:
        if isinstance(request.operation, SmokeInferenceOperation):
            project = self._projects.get(request.operation.project_id)
            if project is None:
                raise ProjectNotFoundError(request.operation.project_id)
            self._ensure_worker_matches_project(project, request.worker)
            graph = project.graph
            validation = self._validator.validate(
                graph, project.execution_backend, project.keras_enabled
            )
            if not validation.valid:
                raise CompilationRejectedError(validation.diagnostics)
            return self._controller.submit(
                request,
                graph=graph,
                compilation_catalog=self._compilation_catalog_provider.get(),
            )

        operation = request.operation
        if not isinstance(operation, ModelExportOperation):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise RuntimeError("Unsupported job operation")
        project_id = operation.project_id
        project = self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        self._ensure_worker_matches_project(project, request.worker)
        validation = self._validator.validate(
            project.graph, project.execution_backend, project.keras_enabled
        )
        if not validation.valid:
            raise CompilationRejectedError(validation.diagnostics)
        capabilities = self._exports.capabilities(project_id)
        capability = next(
            (item for item in capabilities.formats if item.format is operation.format),
            None,
        )
        if capability is None or not capability.available:
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="export_unavailable",
                        severity=DiagnosticSeverity.COMPATIBILITY,
                        message=capability.reason
                        if capability and capability.reason
                        else f"Export format '{operation.format.value}' is unavailable",
                        execution_backend=project.execution_backend,
                        keras_enabled=project.keras_enabled,
                        details={
                            "format": operation.format.value,
                            "reason_code": capability.reason_code
                            if capability
                            else "unknown_format",
                        },
                    )
                ]
            )
        self._validate_export_options(operation.format, operation.options)
        stem = self._resolve_filename_stem(project.name, operation.filename_stem)
        export_operation = operation.model_copy(update={"filename_stem": stem})
        destination_root = (self._settings.export_root / str(project_id)).resolve()
        destination = (
            destination_root / f"{stem}{export_suffix(operation.format, operation.options)}"
        )
        if not destination.is_relative_to(destination_root):
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="invalid_filename",
                        severity=DiagnosticSeverity.ERROR,
                        message="The export filename resolves outside the project export directory",
                    )
                ]
            )
        return self._controller.submit(
            request.model_copy(update={"operation": export_operation}),
            graph=project.graph,
            artifact_destination=destination,
            artifact_filename=destination.name,
            project_revision=project.current_revision,
            revision_checker=lambda: self._project_revision_is_current(
                project_id, project.current_revision
            ),
            compilation_catalog=self._compilation_catalog_provider.get(),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _ensure_worker_matches_project(
        project: ProjectDocument, worker: WorkerConfiguration
    ) -> None:
        execution_backend = project.execution_backend
        keras_enabled = project.keras_enabled
        if worker.backend is not execution_backend:
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="runtime_backend_mismatch",
                        severity=DiagnosticSeverity.COMPATIBILITY,
                        message=(
                            f"This model is persisted for the {execution_backend.value} backend; "
                            f"the worker requested {worker.backend.value}"
                        ),
                        execution_backend=execution_backend,
                        keras_enabled=keras_enabled,
                    )
                ]
            )
        expected_api = RuntimeApi.KERAS if keras_enabled else RuntimeApi.NATIVE
        if worker.api is not expected_api:
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="runtime_api_mismatch",
                        severity=DiagnosticSeverity.COMPATIBILITY,
                        message=(
                            "This model is persisted as a Keras profile and requires the Keras runtime API"
                            if keras_enabled
                            else "This model is persisted as a native profile and requires the native framework API"
                        ),
                        execution_backend=execution_backend,
                        keras_enabled=keras_enabled,
                    )
                ]
            )

    # -------------------------------------------------------------------------
    def _project_revision_is_current(self, project_id: UUID, revision: int) -> bool:
        project = self._projects.get(project_id)
        return project is not None and project.current_revision == revision

    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_filename_stem(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return (sanitized or "model")[:120]

    # -------------------------------------------------------------------------
    @classmethod
    def _resolve_filename_stem(cls, project_name: str, requested: str | None) -> str:
        if requested is None:
            return cls._safe_filename_stem(project_name)
        if any(separator in requested for separator in ("/", "\\")) or "\x00" in requested:
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="invalid_filename",
                        severity=DiagnosticSeverity.ERROR,
                        message="Filename stems cannot contain path separators",
                    )
                ]
            )
        if re.search(r"\.[A-Za-z0-9]+$", requested.strip()):
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="invalid_filename",
                        severity=DiagnosticSeverity.ERROR,
                        message="Provide a filename stem without an extension",
                    )
                ]
            )
        return cls._safe_filename_stem(requested)

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_export_options(export_format: ExportFormat, options: ExportOptions) -> None:
        if options.max_shard_size_gb is not None:
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="invalid_export_options",
                        severity=DiagnosticSeverity.ERROR,
                        message="Weight sharding is not enabled for this export profile",
                        details={"format": export_format.value, "option": "max_shard_size_gb"},
                    )
                ]
            )
        if export_format is not ExportFormat.ONNX and (
            options.opset_version != 18
            or not options.dynamic_batch
            or options.external_data != "auto"
        ):
            raise CompilationRejectedError(
                [
                    ValidationDiagnostic(
                        code="invalid_export_options",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"ONNX options are only valid for the ONNX export format, not '{export_format.value}'",
                        details={"format": export_format.value},
                    )
                ]
            )
