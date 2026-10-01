# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from domain.enums import ExecutionBackend
from domain.runtime import RuntimeApi, WorkerConfiguration

StorageMode = Literal["full", "sampled", "summary_only"]
DataMode = Literal["exact", "sampled", "downsampled"]


###############################################################################
class TensorInspectionSubmit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    project_revision: int = Field(ge=1)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    worker: WorkerConfiguration


###############################################################################
class TensorStatistics(BaseModel):
    mode: DataMode
    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    median: float | None = None
    zero_proportion: float | None = Field(default=None, ge=0, le=1)
    nan_count: int = Field(default=0, ge=0)
    infinite_count: int = Field(default=0, ge=0)


###############################################################################
class TensorHistogram(BaseModel):
    mode: DataMode
    bins: list[float] = Field(default_factory=lambda: list[float]())
    counts: list[int] = Field(default_factory=lambda: list[int]())


###############################################################################
class TensorDescriptor(BaseModel):
    id: str
    storage_id: str
    node_id: str
    direction: Literal["input", "output"]
    port_id: str
    item_index: int | None = None
    shape: list[int]
    dtype: str
    element_count: int = Field(ge=0)
    storage_mode: StorageMode
    value_count: int = Field(ge=0)
    statistics: TensorStatistics | None = None
    histogram: TensorHistogram | None = None


###############################################################################
class TensorInspectionNode(BaseModel):
    node_id: str
    node_name: str
    node_type: str
    topological_index: int
    depth: int = Field(ge=0)
    predecessors: list[str] = Field(default_factory=list)
    successors: list[str] = Field(default_factory=list)
    inspectable: bool = True
    unavailable_reason: str | None = None
    inputs: list[TensorDescriptor] = Field(default_factory=lambda: list[TensorDescriptor]())
    outputs: list[TensorDescriptor] = Field(default_factory=lambda: list[TensorDescriptor]())


###############################################################################
class TensorInspectionSession(BaseModel):
    id: str
    project_id: UUID
    project_revision: int
    execution_backend: ExecutionBackend
    runtime_api: RuntimeApi
    worker: WorkerConfiguration
    seed: int
    weight_source: Literal["initialized_untrained"] = "initialized_untrained"
    input_description: str
    compiler_mode: Literal["normal", "bypassed_for_inspection"]
    created_at: datetime
    expires_at: datetime
    nodes: list[TensorInspectionNode]


###############################################################################
class TensorInspectionViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_id: str
    fixed_indices: dict[int, int] = Field(default_factory=lambda: dict[int, int]())
    display_axes: list[int] = Field(default_factory=lambda: list[int](), max_length=2)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=256, ge=1, le=4096)


###############################################################################
class TensorInspectionViewResponse(BaseModel):
    tensor_id: str
    shape: list[int]
    dtype: str
    display_axes: list[int]
    fixed_indices: dict[int, int]
    values: Any
    view_shape: list[int]
    mode: DataMode
    total_values: int = Field(ge=0)
