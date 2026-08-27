from __future__ import annotations

from configurations.settings import Settings
from domain.compilation_capabilities import get_compilation_capabilities
from domain.compilation_catalog import CompilationCatalogProvider, get_compilation_catalog_provider
from domain.enums import ExecutionBackend
from domain.graph import GRAPH_SCHEMA_VERSION
from domain.metadata import (
    ExecutionBackendStatus,
    HealthResponse,
    MetadataConfiguration,
    MetadataResponse,
)
from domain.runtime import RuntimeApi, RuntimeApiStatus


###############################################################################
class RuntimeService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        settings: Settings,
        backend_statuses: dict[ExecutionBackend, ExecutionBackendStatus],
        compilation_catalog_provider: CompilationCatalogProvider | None = None,
    ) -> None:
        self._settings = settings
        self._backend_statuses = backend_statuses
        self._compilation_catalog_provider = compilation_catalog_provider or get_compilation_catalog_provider()

    # -------------------------------------------------------------------------
    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    # -------------------------------------------------------------------------
    def metadata(self) -> MetadataResponse:
        catalog = self._compilation_catalog_provider.get()
        backend_statuses = dict(self._backend_statuses)
        runtime_apis = [
            RuntimeApiStatus(
                id=api,
                available=any(
                    api_status.available
                    for status in self._backend_statuses.values()
                    for api_status in status.runtime_apis
                    if api_status.id is api
                ),
                reason_code=next(
                    (
                        api_status.reason_code
                        for status in self._backend_statuses.values()
                        for api_status in status.runtime_apis
                        if api_status.id is api and not api_status.available and api_status.reason_code
                    ),
                    None,
                ),
            )
            for api in RuntimeApi
        ]
        return MetadataResponse(
            name=self._settings.app_name,
            version=self._settings.app_version,
            schema_version=GRAPH_SCHEMA_VERSION,
            runtime_apis=runtime_apis,
            execution_backends=[
                ExecutionBackendStatus(
                    id=backend,
                    installed=status.installed,
                    version=status.version,
                    available=status.available,
                    reason_code=status.reason_code,
                    accelerator=status.accelerator,
                    device_name=status.device_name,
                    devices=status.devices,
                    runtime_apis=status.runtime_apis,
                    native_compilers=status.native_compilers,
                )
                for backend, status in backend_statuses.items()
            ],
            configuration=MetadataConfiguration(
                compilation_capabilities=get_compilation_capabilities(catalog, backend_statuses),
            ),
        )
