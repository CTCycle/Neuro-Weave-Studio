from __future__ import annotations

from collections.abc import Sequence

from configurations.settings import Settings
from domain.enums import KerasBackend, TargetFramework
from domain.graph import GRAPH_SCHEMA_VERSION
from domain.metadata import HealthResponse, MetadataConfiguration, MetadataResponse


class RuntimeService:
    def __init__(self, settings: Settings, available_keras_backends: Sequence[KerasBackend]) -> None:
        self._settings = settings
        self._available_keras_backends = list(available_keras_backends)

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    def metadata(self) -> MetadataResponse:
        return MetadataResponse(
            name=self._settings.app_name,
            version=self._settings.app_version,
            schema_version=GRAPH_SCHEMA_VERSION,
            frameworks=list(TargetFramework),
            available_keras_backends=self._available_keras_backends,
            configuration=MetadataConfiguration(),
        )
