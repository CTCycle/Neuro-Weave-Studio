from __future__ import annotations

from typing import cast

from fastapi import Depends, Request

from common.catalogs.loader import NodeCatalog
from configurations.settings import Settings
from domain.enums import ExecutionBackend
from domain.metadata import ExecutionBackendStatus
from models.runtime_controller import RuntimeController
from repositories.contracts import UnitOfWorkFactory
from repositories.session import RepositoryContext
from services.catalog_service import CatalogService
from services.compilation_service import CompilationService
from services.job_service import JobService
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
def get_runtime_controller(request: Request) -> RuntimeController:
    return cast(RuntimeController, request.app.state.runtime_controller)

###############################################################################
def get_backend_statuses(request: Request) -> dict[ExecutionBackend, ExecutionBackendStatus]:
    return cast(dict[ExecutionBackend, ExecutionBackendStatus], request.app.state.backend_statuses)

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
    statuses = cast(dict[ExecutionBackend, ExecutionBackendStatus], request.app.state.backend_statuses)
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
    settings: Settings = Depends(get_settings),
) -> ExportService:
    return ExportService(
        project_service=project_service,
        compilation_service=compilation_service,
        validator=validator,
        uow_factory=uow_factory,
        settings=settings,
    )

###############################################################################
def get_job_service(
    catalog: NodeCatalog = Depends(get_catalog),
    project_service: ProjectService = Depends(get_project_service),
    controller: RuntimeController = Depends(get_runtime_controller),
    settings: Settings = Depends(get_settings),
) -> JobService:
    return JobService(catalog, project_service, controller, settings)

###############################################################################
