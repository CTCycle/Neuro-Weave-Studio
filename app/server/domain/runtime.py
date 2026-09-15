from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain.enums import ExecutionBackend


###############################################################################
class RuntimeApi(StrEnum):
    """The model API used inside an isolated worker."""

    KERAS = "keras"
    NATIVE = "native"

###############################################################################
class RuntimeDiscoveryState(StrEnum):
    """Lifecycle state for optional framework capability discovery."""

    UNCHECKED = "unchecked"
    CHECKING = "checking"
    VERIFIED = "verified"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"

###############################################################################
class MemoryPolicyKind(StrEnum):
    DEFAULT = "default"
    TENSORFLOW_GROWTH = "tensorflow_growth"
    TENSORFLOW_LIMIT_MB = "tensorflow_limit_mb"
    JAX_NO_PREALLOCATION = "jax_no_preallocation"
    JAX_PREALLOCATE_FRACTION = "jax_preallocate_fraction"
    JAX_PLATFORM_ALLOCATOR = "jax_platform_allocator"
    PYTORCH_FRACTION = "pytorch_fraction"

###############################################################################
class MemoryPolicy(BaseModel):
    """A startup-only allocator policy; values are never silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: MemoryPolicyKind = MemoryPolicyKind.DEFAULT
    limit_mb: int | None = Field(default=None, ge=1, le=2_000_000)
    fraction: float | None = Field(default=None, gt=0, le=1)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_parameters(self) -> MemoryPolicy:
        needs_limit = self.kind is MemoryPolicyKind.TENSORFLOW_LIMIT_MB
        needs_fraction = self.kind in {
            MemoryPolicyKind.JAX_PREALLOCATE_FRACTION,
            MemoryPolicyKind.PYTORCH_FRACTION,
        }
        if needs_limit and self.limit_mb is None:
            raise ValueError("tensorflow_limit_mb requires limit_mb")
        if not needs_limit and self.limit_mb is not None:
            raise ValueError("limit_mb is only valid for tensorflow_limit_mb")
        if needs_fraction and self.fraction is None:
            raise ValueError(f"{self.kind.value} requires fraction")
        if not needs_fraction and self.fraction is not None:
            raise ValueError("fraction is only valid for a fraction memory policy")
        return self

###############################################################################
class WorkerConfiguration(BaseModel):
    """Immutable process-start configuration shared by every ML operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api: RuntimeApi = RuntimeApi.KERAS
    backend: ExecutionBackend
    device_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_.:-]*$")
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    precision: Literal["float32", "mixed_float16", "mixed_bfloat16"] = "float32"

###############################################################################
class RuntimeDevice(BaseModel):
    device_id: str
    lease_key: str
    kind: str
    index: int | None = None
    name: str
    accelerator: str
    available: bool = True
    total_memory_bytes: int | None = Field(default=None, ge=0)
    supported_memory_policies: list[MemoryPolicyKind] = Field(default_factory=lambda: list[MemoryPolicyKind]())

###############################################################################
class RuntimeApiStatus(BaseModel):
    id: RuntimeApi
    available: bool
    reason_code: str | None = None
    state: RuntimeDiscoveryState = RuntimeDiscoveryState.UNCHECKED
    source: Literal["snapshot", "live", "none"] = "none"
    checked_at: datetime | None = None

###############################################################################
class RuntimeDiscoveryRequest(BaseModel):
    backend: ExecutionBackend
    api: RuntimeApi = RuntimeApi.KERAS
    force_refresh: bool = False

###############################################################################
class RuntimeDiscoveryResponse(BaseModel):
    backend: ExecutionBackend
    api: RuntimeApi
    state: RuntimeDiscoveryState
    available: bool = False
    reason_code: str | None = None
    source: Literal["snapshot", "live", "none"] = "none"
    checked_at: datetime | None = None
    message: str | None = None

###############################################################################
class DeviceLeaseStatus(BaseModel):
    device_id: str
    state: Literal[
        "idle",
        "starting",
        "running",
        "stop_requested",
        "stopping",
        "failed",
    ] = "idle"
    job_id: str | None = None
    pid: int | None = None
