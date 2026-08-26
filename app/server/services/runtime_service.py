from __future__ import annotations

from configurations.settings import Settings
from domain.compilation_catalog import get_compilation_catalog
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
    ) -> None:
        self._settings = settings
        self._backend_statuses = backend_statuses

    # -------------------------------------------------------------------------
    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    # -------------------------------------------------------------------------
    def metadata(self) -> MetadataResponse:
        catalog = get_compilation_catalog()
        return MetadataResponse(
            name=self._settings.app_name,
            version=self._settings.app_version,
            schema_version=GRAPH_SCHEMA_VERSION,
            runtime_apis=[
                RuntimeApiStatus(id=RuntimeApi.KERAS, available=True),
                RuntimeApiStatus(
                    id=RuntimeApi.NATIVE,
                    available=False,
                    reason_code="not_implemented",
                ),
            ],
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
                )
                for backend, status in self._backend_statuses.items()
            ],
            configuration=MetadataConfiguration(
                compilation_catalog=catalog,
                optimizers=[item.id for item in catalog.optimizers],
                losses=[item.id for item in catalog.losses],
                metrics=[item.id for item in catalog.metrics],
                initializers=catalog.initializers,
                regularizers=catalog.regularizers,
            ),
        )
