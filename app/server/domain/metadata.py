from __future__ import annotations

from pydantic import BaseModel, Field

from domain.compilation_capabilities import CompilationCapabilities
from domain.enums import ExecutionBackend
from domain.runtime import RuntimeApiStatus, RuntimeDevice


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
    runtime_apis: list[RuntimeApiStatus] = Field(default_factory=list)
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
    devices: list[RuntimeDevice] = Field(default_factory=list)
    runtime_apis: list[RuntimeApiStatus] = Field(default_factory=list)
    native_compilers: dict[str, bool] = Field(default_factory=dict)
