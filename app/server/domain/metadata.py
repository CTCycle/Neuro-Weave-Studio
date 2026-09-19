from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from domain.compilation_capabilities import CompilationCapabilities
from domain.enums import ExecutionBackend
from domain.runtime import RuntimeApiStatus, RuntimeDevice, RuntimeDiscoveryState

###############################################################################
class HealthResponse(BaseModel):
    status: str

###############################################################################
class MetadataConfiguration(BaseModel):
    compilation_capabilities: CompilationCapabilities

###############################################################################
class MetadataResponse(BaseModel):
    name: str
    version: str
    schema_version: int
    runtime_apis: list[RuntimeApiStatus] = Field(default_factory=lambda: list[RuntimeApiStatus]())
    execution_backends: list[ExecutionBackendStatus]
    configuration: MetadataConfiguration

###############################################################################
class ExecutionBackendStatus(BaseModel):
    id: ExecutionBackend
    installed: bool = False
    version: str | None = None
    available: bool
    reason_code: str | None = None
    accelerator: str | None = None
    device_name: str | None = None
    devices: list[RuntimeDevice] = Field(default_factory=lambda: list[RuntimeDevice]())
    runtime_apis: list[RuntimeApiStatus] = Field(default_factory=lambda: list[RuntimeApiStatus]())
    native_compilers: dict[str, bool] = Field(default_factory=dict)
    discovery_state: RuntimeDiscoveryState = RuntimeDiscoveryState.UNCHECKED
    discovery_source: Literal["snapshot", "live", "none"] = "none"
    last_checked_at: datetime | None = None
