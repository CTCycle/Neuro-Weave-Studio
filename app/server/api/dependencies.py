from __future__ import annotations

from typing import cast

from fastapi import Depends, Request

from common.catalogs.loader import NodeCatalog
from configurations.settings import Settings
from repositories.contracts import UnitOfWorkFactory
from repositories.session import RepositoryContext
from services.catalog_service import CatalogService
from services.compilation_service import CompilationService
from services.export_service import ExportService
from services.job_service import JobService
from services.keras_import_service import KerasImportService
from services.project_service import ProjectService
from services.runtime_controller import RuntimeController
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
def get_validation_service(
    catalog: NodeCatalog = Depends(get_catalog),
) -> ValidationService:
    return ValidationService(catalog)

###############################################################################
def get_catalog_service(catalog: NodeCatalog = Depends(get_catalog)) -> CatalogService:
    return CatalogService(catalog)

###############################################################################
def get_runtime_service(
    settings: Settings = Depends(get_settings),
    controller: RuntimeController = Depends(get_runtime_controller),
) -> RuntimeService:
    return RuntimeService(settings, controller.backend_statuses)

###############################################################################
def get_project_service(
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
    validator: ValidationService = Depends(get_validation_service),
    controller: RuntimeController = Depends(get_runtime_controller),
) -> ProjectService:
    return ProjectService(uow_factory, validator, controller.backend_statuses)

###############################################################################
def get_keras_import_service(
    project_service: ProjectService = Depends(get_project_service),
) -> KerasImportService:
    return KerasImportService(project_service)

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
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
    runtime_controller: RuntimeController = Depends(get_runtime_controller),
    validation_service: ValidationService = Depends(get_validation_service),
) -> ExportService:
    return ExportService(
        project_service=project_service,
        compilation_service=compilation_service,
        uow_factory=uow_factory,
        runtime_controller=runtime_controller,
        validation_service=validation_service,
    )

###############################################################################
def get_job_service(
    project_service: ProjectService = Depends(get_project_service),
    controller: RuntimeController = Depends(get_runtime_controller),
    settings: Settings = Depends(get_settings),
    validator: ValidationService = Depends(get_validation_service),
    exports: ExportService = Depends(get_export_service),
) -> JobService:
    return JobService(project_service, controller, settings, validator, exports)

###############################################################################
