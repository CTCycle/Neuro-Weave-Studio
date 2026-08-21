from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from domain.graph import ModelGraph
from domain.runtime import DeviceLeaseStatus, WorkerConfiguration

###############################################################################
class JobOperation(StrEnum):
    SMOKE_INFERENCE = "smoke_inference"
    KERAS_EXPORT = "keras_export"

###############################################################################
class JobState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    TERMINATED = "terminated"
    FAILED = "failed"

###############################################################################
class SmokeInferenceOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[JobOperation.SMOKE_INFERENCE] = JobOperation.SMOKE_INFERENCE
    project_id: UUID
    graph: ModelGraph

###############################################################################
class KerasExportOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[JobOperation.KERAS_EXPORT] = JobOperation.KERAS_EXPORT
    project_id: UUID

###############################################################################
JobOperationRequest = Annotated[
    SmokeInferenceOperation | KerasExportOperation,
    Field(discriminator="type"),
]

###############################################################################
class JobSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: JobOperationRequest
    worker: WorkerConfiguration

###############################################################################
class JobProgress(BaseModel):
    phase: str = "queued"
    message: str = "Worker is starting"
    percent: int | None = Field(default=None, ge=0, le=100)

###############################################################################
class JobDiagnostics(BaseModel):
    process_rss_bytes: int | None = Field(default=None, ge=0)
    framework_allocated_bytes: int | None = Field(default=None, ge=0)
    framework_reserved_bytes: int | None = Field(default=None, ge=0)
    framework_peak_bytes: int | None = Field(default=None, ge=0)
    device_used_bytes: int | None = Field(default=None, ge=0)
    device_total_bytes: int | None = Field(default=None, ge=0)

###############################################################################
class JobError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

###############################################################################
class JobResponse(BaseModel):
    id: str
    operation: JobOperation
    state: JobState
    worker: WorkerConfiguration
    device_lease: DeviceLeaseStatus
    pid: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    finished_at: datetime | None = None
    exit_code: int | None = None
    cancellation_requested: bool = False
    forced_termination: bool = False
    restart_reason: str = "new_job"
    progress: JobProgress = Field(default_factory=JobProgress)
    diagnostics: JobDiagnostics = Field(default_factory=JobDiagnostics)
    result: dict[str, Any] | None = None
    artifact_available: bool = False
    error: JobError | None = None
