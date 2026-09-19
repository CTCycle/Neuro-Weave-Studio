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
from domain.runtime import RuntimeApi, RuntimeApiStatus, RuntimeDiscoveryState

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
        self._compilation_catalog_provider = (
            compilation_catalog_provider or get_compilation_catalog_provider()
        )

    # -------------------------------------------------------------------------
    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    # -------------------------------------------------------------------------
    def metadata(self) -> MetadataResponse:
        catalog = self._compilation_catalog_provider.get()
        backend_statuses = dict(self._backend_statuses)
        runtime_apis: list[RuntimeApiStatus] = []
        for api in RuntimeApi:
            matching = [
                api_status
                for status in self._backend_statuses.values()
                for api_status in status.runtime_apis
                if api_status.id is api
            ]
            available = any(item.available for item in matching)
            checking = any(item.state.value == "checking" for item in matching)
            terminal = [item for item in matching if item.state.value in {"verified", "unavailable", "failed"}]
            state = (
                RuntimeDiscoveryState.CHECKING
                if checking
                else RuntimeDiscoveryState.VERIFIED
                if available
                else RuntimeDiscoveryState.FAILED
                if any(item.state is RuntimeDiscoveryState.FAILED for item in terminal)
                else RuntimeDiscoveryState.UNAVAILABLE
                if terminal and len(terminal) == len(matching)
                else RuntimeDiscoveryState.UNCHECKED
            )
            source = "live" if any(item.source == "live" for item in matching) else "snapshot" if any(item.source == "snapshot" for item in matching) else "none"
            checked_at = max((item.checked_at for item in matching if item.checked_at is not None), default=None)
            runtime_apis.append(RuntimeApiStatus(
                id=api,
                available=available,
                reason_code=None if available else next((item.reason_code for item in matching if not item.available and item.reason_code), None),
                state=state,
                source=source,
                checked_at=checked_at,
            ))
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
                    discovery_state=status.discovery_state,
                    discovery_source=status.discovery_source,
                    last_checked_at=status.last_checked_at,
                )
                for backend, status in backend_statuses.items()
            ],
            configuration=MetadataConfiguration(
                compilation_capabilities=get_compilation_capabilities(catalog, backend_statuses),
            ),
        )
