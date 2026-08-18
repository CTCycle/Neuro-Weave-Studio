from __future__ import annotations

from collections.abc import Mapping

from configurations.settings import Settings
from domain.enums import CodeTarget, ExecutionBackend
from domain.graph import GRAPH_SCHEMA_VERSION
from domain.metadata import (
    ExecutionBackendStatus,
    HealthResponse,
    MetadataConfiguration,
    MetadataResponse,
)

###############################################################################
class RuntimeService:

    # -------------------------------------------------------------------------
    def __init__(self, settings: Settings, backend_statuses: Mapping[ExecutionBackend, Mapping[str, object]]) -> None:
        self._settings = settings
        self._backend_statuses = backend_statuses

    # -------------------------------------------------------------------------
    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    # -------------------------------------------------------------------------
    def metadata(self) -> MetadataResponse:
        return MetadataResponse(
            name=self._settings.app_name,
            version=self._settings.app_version,
            schema_version=GRAPH_SCHEMA_VERSION,
            code_targets=list(CodeTarget),
            execution_backends=[
                ExecutionBackendStatus(
                    id=backend,
                    installed=bool(status.get("installed")),
                    version=(str(status.get("version")) if status.get("version") else None),
                    available=bool(status.get("available")),
                    reason_code=(str(status.get("reason_code")) if status.get("reason_code") else None),
                    accelerator=(str(status.get("accelerator")) if status.get("accelerator") else None),
                    device_name=(str(status.get("device_name")) if status.get("device_name") else None),
                )
                for backend, status in self._backend_statuses.items()
            ],
            configuration=MetadataConfiguration(),
        )
