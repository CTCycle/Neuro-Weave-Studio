from __future__ import annotations

from typing import cast

from fastapi import Depends, Request

from common.catalogs.loader import NodeCatalog
from configurations.settings import Settings
from domain.enums import ExecutionBackend
from models.keras_backend_runner import KerasBackendRunner
from repositories.contracts import UnitOfWorkFactory
from repositories.session import RepositoryContext
from services.catalog_service import CatalogService
from services.compilation_service import CompilationService
from services.execution_service import ExecutionService
from services.export_service import ExportService
from services.project_service import ProjectService
from services.runtime_service import RuntimeService
from services.validation_service import ValidationService


###############################################################################
def get_catalog(request: Request) -> NodeCatalog:
    return cast(NodeCatalog, request.app.state.catalog)

###############################################################################
def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)

###############################################################################
def get_repository_context(request: Request) -> RepositoryContext:
    return cast(RepositoryContext, request.app.state.repository_context)

###############################################################################
def get_uow_factory(
    repository_context: RepositoryContext = Depends(get_repository_context),
) -> UnitOfWorkFactory:
    return repository_context.new_unit_of_work

###############################################################################
def get_backend_runner(request: Request) -> KerasBackendRunner:
    return cast(KerasBackendRunner, request.app.state.backend_runner)

###############################################################################
def get_backend_statuses(request: Request) -> dict[ExecutionBackend, dict[str, object]]:
    return cast(dict[ExecutionBackend, dict[str, object]], request.app.state.backend_statuses)

###############################################################################
def get_validation_service(
    catalog: NodeCatalog = Depends(get_catalog),
) -> ValidationService:
    return ValidationService(catalog)

###############################################################################
def get_catalog_service(catalog: NodeCatalog = Depends(get_catalog)) -> CatalogService:
    return CatalogService(catalog)

###############################################################################
def get_runtime_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> RuntimeService:
    statuses = cast(dict[ExecutionBackend, dict[str, object]], request.app.state.backend_statuses)
    return RuntimeService(settings, statuses)

###############################################################################
def get_project_service(
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
    validator: ValidationService = Depends(get_validation_service),
) -> ProjectService:
    return ProjectService(uow_factory, validator)

###############################################################################
def get_compilation_service(
    catalog: NodeCatalog = Depends(get_catalog),
    validator: ValidationService = Depends(get_validation_service),
) -> CompilationService:
    return CompilationService(catalog, validator)

###############################################################################
def get_export_service(
    project_service: ProjectService = Depends(get_project_service),
    compilation_service: CompilationService = Depends(get_compilation_service),
    validator: ValidationService = Depends(get_validation_service),
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
    backend_runner: KerasBackendRunner = Depends(get_backend_runner),
    backend_statuses: dict[ExecutionBackend, dict[str, object]] = Depends(get_backend_statuses),
    settings: Settings = Depends(get_settings),
) -> ExportService:
    return ExportService(
        project_service=project_service,
        compilation_service=compilation_service,
        validator=validator,
        uow_factory=uow_factory,
        backend_runner=backend_runner,
        available_execution_backends=tuple(
            backend for backend, status in backend_statuses.items() if status.get("available")
        ),
        settings=settings,
    )

###############################################################################
def get_execution_service(
    catalog: NodeCatalog = Depends(get_catalog),
    backend_runner: KerasBackendRunner = Depends(get_backend_runner),
    backend_statuses: dict[ExecutionBackend, dict[str, object]] = Depends(get_backend_statuses),
) -> ExecutionService:
    return ExecutionService(catalog, backend_runner, backend_statuses)
