from __future__ import annotations

from uuid import UUID

from common.timestamps import utc_now
from configurations.settings import Settings
from domain.compilation import CompilationResult
from domain.enums import CodeTarget
from domain.exports import SourceExportResponse
from repositories.contracts import UnitOfWorkFactory
from services.compilation_service import CompilationService
from services.errors import (
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
        settings: Settings,
    ) -> None:
        self._projects = project_service
        self._compilation = compilation_service
        self._validator = validator
        self._uow_factory = uow_factory
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
