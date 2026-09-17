from __future__ import annotations

from typing import cast

from fastapi import Depends, Request

from common.catalogs.loader import NodeCatalog
from configurations.settings import Settings
from domain.compilation_catalog import (
    CompilationCatalogProvider,
    PersistentCompilationCatalogProvider,
)
from repositories.contracts import UnitOfWorkFactory
from repositories.session import RepositoryContext
from services.catalog_service import CatalogService
from services.compilation_service import CompilationService
from services.connection_validation_service import ConnectionValidationService
from services.custom_node_service import CustomNodeService
from services.dataset_service import DatasetService
from services.export_service import ExportService
from services.filesystem_service import FilesystemService
from services.job_service import JobService
from services.keras_import_service import KerasImportService
from services.metric_service import MetricService
from services.project_service import ProjectService
from services.runtime_controller import RuntimeController
from services.runtime_service import RuntimeService
from services.tensor_inspection_service import TensorInspectionService
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
def get_compilation_catalog_provider(
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
) -> CompilationCatalogProvider:
    return PersistentCompilationCatalogProvider(uow_factory)

###############################################################################
def get_validation_service(
    catalog: NodeCatalog = Depends(get_catalog),
    compilation_catalog_provider: CompilationCatalogProvider = Depends(get_compilation_catalog_provider),
) -> ValidationService:
    return ValidationService(catalog, compilation_catalog_provider)

###############################################################################
def get_connection_validation_service(
    catalog: NodeCatalog = Depends(get_catalog),
    validator: ValidationService = Depends(get_validation_service),
) -> ConnectionValidationService:
    return ConnectionValidationService(catalog, validator)


###############################################################################
def get_metric_service(
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
    compilation_catalog_provider: CompilationCatalogProvider = Depends(get_compilation_catalog_provider),
) -> MetricService:
    return MetricService(uow_factory, compilation_catalog_provider)

###############################################################################
def get_catalog_service(catalog: NodeCatalog = Depends(get_catalog)) -> CatalogService:
    return CatalogService(catalog)

###############################################################################
def get_custom_node_service(
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
    catalog: NodeCatalog = Depends(get_catalog),
) -> CustomNodeService:
    return CustomNodeService(uow_factory, catalog)

###############################################################################
def get_filesystem_service() -> FilesystemService:
    return FilesystemService()

###############################################################################
def get_dataset_service(
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
) -> DatasetService:
    return DatasetService(uow_factory)

###############################################################################
def get_runtime_service(
    settings: Settings = Depends(get_settings),
    controller: RuntimeController = Depends(get_runtime_controller),
    compilation_catalog_provider: CompilationCatalogProvider = Depends(get_compilation_catalog_provider),
) -> RuntimeService:
    return RuntimeService(settings, controller.backend_statuses, compilation_catalog_provider)

###############################################################################
def get_project_service(
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
    validator: ValidationService = Depends(get_validation_service),
    controller: RuntimeController = Depends(get_runtime_controller),
) -> ProjectService:
    return ProjectService(uow_factory, validator)

###############################################################################
def get_keras_import_service(
    project_service: ProjectService = Depends(get_project_service),
) -> KerasImportService:
    return KerasImportService(project_service)

###############################################################################
def get_compilation_service(
    catalog: NodeCatalog = Depends(get_catalog),
    validator: ValidationService = Depends(get_validation_service),
    compilation_catalog_provider: CompilationCatalogProvider = Depends(get_compilation_catalog_provider),
) -> CompilationService:
    return CompilationService(catalog, validator, compilation_catalog_provider)

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
    compilation_catalog_provider: CompilationCatalogProvider = Depends(get_compilation_catalog_provider),
) -> JobService:
    return JobService(project_service, controller, settings, validator, exports, compilation_catalog_provider)

###############################################################################
def get_tensor_inspection_service(
    settings: Settings = Depends(get_settings),
) -> TensorInspectionService:
    return TensorInspectionService(settings)


###############################################################################
