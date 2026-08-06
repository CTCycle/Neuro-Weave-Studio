from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from common.timestamps import utc_now
from configurations.settings import Settings
from domain.compilation import CompilationResult
from domain.enums import CodeTarget, ExecutionBackend
from domain.exports import SourceExportResponse
from models.keras_backend_runner import KerasBackendRunner
from repositories.contracts import UnitOfWorkFactory
from services.compilation_service import CompilationService
from services.errors import (
    BinaryExportUnavailableError,
    CompilationRejectedError,
    ProjectNotFoundError,
    RevisionConflictError,
)
from services.project_service import ProjectService
from services.validation_service import ValidationService


###############################################################################
class ExportService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        project_service: ProjectService,
        compilation_service: CompilationService,
        validator: ValidationService,
        uow_factory: UnitOfWorkFactory,
        backend_runner: KerasBackendRunner,
        available_execution_backends: Sequence[ExecutionBackend],
        settings: Settings,
    ) -> None:
        self._projects = project_service
        self._compilation = compilation_service
        self._validator = validator
        self._uow_factory = uow_factory
        self._backend_runner = backend_runner
        self._available_execution_backends = tuple(available_execution_backends)
        self._settings = settings

    # -------------------------------------------------------------------------
    def source(self, project_id: UUID, code_target: CodeTarget) -> SourceExportResponse:
        project = self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        result: CompilationResult = self._compilation.compile(project.graph, code_target)
        if not result.valid or result.code is None:
            raise CompilationRejectedError(result.diagnostics)

        with self._uow_factory() as uow:
            current = uow.projects.get_metadata(project_id)
            if current is None:
                raise ProjectNotFoundError(project_id)
            if current.current_revision != project.current_revision:
                raise RevisionConflictError(
                    "The project changed while source code was being generated"
                )
            uow.code_snapshots.upsert(
                project_id,
                revision=current.current_revision,
                code_target=code_target,
                source_code=result.code,
                timestamp=utc_now(),
            )
            uow.commit()
        return SourceExportResponse(code_target=code_target, code=result.code)

    # -------------------------------------------------------------------------
    def keras_model(self, project_id: UUID, execution_backend: ExecutionBackend) -> tuple[Path, str]:
        project = self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        validation = self._validator.validate(project.graph, CodeTarget.KERAS)
        if not validation.valid:
            raise CompilationRejectedError(validation.diagnostics)
        if execution_backend not in self._available_execution_backends:
            raise BinaryExportUnavailableError(execution_backend)
        destination = (
            self._settings.export_root
            / str(project_id)
            / f"{self._safe_filename_stem(project.name)}.keras"
        )
        try:
            self._backend_runner.export(project.graph, destination, execution_backend)
        except RuntimeError as exc:
            raise BinaryExportUnavailableError(execution_backend) from exc
        return destination, destination.name

    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_filename_stem(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return (sanitized or "model")[:120]
