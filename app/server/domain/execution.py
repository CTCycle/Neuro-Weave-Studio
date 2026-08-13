from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from domain.enums import ExecutionBackend
from domain.graph import ModelGraph

###############################################################################
class SmokeTestRequest(BaseModel):
    """A graph and backend for an inference-only model smoke test."""

    graph: ModelGraph
    execution_backend: ExecutionBackend

###############################################################################
class TensorExecutionInfo(BaseModel):
    node_id: str
    shape: list[int]
    dtype: str

###############################################################################
class SmokeTestResponse(BaseModel):
    status: Literal["succeeded"] = "succeeded"
    execution_backend: ExecutionBackend
    accelerator: str
    device_name: str
    inputs: list[TensorExecutionInfo] = Field(default_factory=lambda: list[TensorExecutionInfo]())
    outputs: list[TensorExecutionInfo] = Field(default_factory=lambda: list[TensorExecutionInfo]())
    parameter_count: int
    duration_ms: float
