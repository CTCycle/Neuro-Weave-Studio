from __future__ import annotations

from pydantic import BaseModel, Field

from domain.compilation_catalog import CompilationCatalog, get_compilation_catalog
from domain.enums import ExecutionBackend
from domain.runtime import RuntimeApiStatus, RuntimeDevice


###############################################################################
class HealthResponse(BaseModel):
    status: str

###############################################################################
class MetadataConfiguration(BaseModel):
    compilation_catalog: CompilationCatalog = Field(default_factory=get_compilation_catalog)
    optimizers: list[str] = Field(default_factory=lambda: [item.id for item in get_compilation_catalog().optimizers])
    losses: list[str] = Field(default_factory=lambda: [item.id for item in get_compilation_catalog().losses])
    metrics: list[str] = Field(default_factory=lambda: [item.id for item in get_compilation_catalog().metrics])
    initializers: list[str] = Field(default_factory=lambda: get_compilation_catalog().initializers)
    regularizers: list[str] = Field(default_factory=lambda: get_compilation_catalog().regularizers)

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
