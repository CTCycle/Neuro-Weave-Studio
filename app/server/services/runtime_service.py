from __future__ import annotations

from collections.abc import Mapping

from configurations.settings import Settings
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
    def __init__(self, settings: Settings, backend_statuses: Mapping[ExecutionBackend, ExecutionBackendStatus | Mapping[str, object]]) -> None:
        self._settings = settings
        self._backend_statuses = {
            backend: status
            if isinstance(status, ExecutionBackendStatus)
            else ExecutionBackendStatus.model_validate({"id": backend, **dict(status)})
            for backend, status in backend_statuses.items()
        }

    # -------------------------------------------------------------------------
    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    # -------------------------------------------------------------------------
    def metadata(self) -> MetadataResponse:
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
            configuration=MetadataConfiguration(),
        )
