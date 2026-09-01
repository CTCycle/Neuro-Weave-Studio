from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.timestamps import utc_now
from domain.enums import ExecutionBackend, ValidationStatus
from domain.tensor import TensorSpec

GRAPH_SCHEMA_VERSION = 4


###############################################################################
class Position(BaseModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


###############################################################################
class NodeInstance(BaseModel):
    id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    definition_id: str
    definition_version: int = Field(default=1, ge=1)
    label: str | None = None
    position: Position
    parameters: dict[str, Any] = Field(default_factory=dict)
    locked: bool = False


###############################################################################
class EdgeInstance(BaseModel):
    id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str


###############################################################################
class OutputBinding(BaseModel):
    """A persisted, explicit model-output port selection."""

    node_id: str = Field(min_length=1, max_length=120)
    port_id: str = Field(min_length=1, max_length=80)


###############################################################################
class KerasCompilationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: str = "adam"
    optimizer_parameters: dict[str, Any] = Field(default_factory=lambda: {"learning_rate": 0.001})
    losses: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, list[str]] = Field(default_factory=dict)
    default_initializer: str = "glorot_uniform"
    default_regularizer: str = "none"
    default_constraint: str = "none"
    mixed_precision: bool = False
    random_seed: int | None = 42


###############################################################################
class TensorFlowNativeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jit_compile: bool = False


###############################################################################
class PyTorchNativeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: str = "adam"
    optimizer_parameters: dict[str, Any] = Field(default_factory=lambda: {"learning_rate": 0.001})
    losses: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, list[str]] = Field(default_factory=dict)
    compile_enabled: bool = False
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False


###############################################################################
class JaxNativeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jit_enabled: bool = False


###############################################################################
class NativeCompilationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensorflow: TensorFlowNativeConfiguration = Field(default_factory=TensorFlowNativeConfiguration)
    pytorch: PyTorchNativeConfiguration = Field(default_factory=PyTorchNativeConfiguration)
    jax: JaxNativeConfiguration = Field(default_factory=JaxNativeConfiguration)


###############################################################################
class ModelConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keras: KerasCompilationConfiguration = Field(default_factory=KerasCompilationConfiguration)
    native: NativeCompilationConfiguration = Field(default_factory=NativeCompilationConfiguration)


###############################################################################
class ExportSettings(BaseModel):
    include_comments: bool = True
    include_shape_assertions: bool = True
    preferred_binary_format: str = "keras"


###############################################################################
class ModelGraph(BaseModel):
    schema_version: Literal[4] = GRAPH_SCHEMA_VERSION
    nodes: list[NodeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    edges: list[EdgeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    output_bindings: list[OutputBinding] = Field(default_factory=list)
    model_configuration: ModelConfiguration = Field(default_factory=ModelConfiguration)
    export_settings: ExportSettings = Field(default_factory=ExportSettings)
    future_training_configuration: dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def unique_ids(self) -> ModelGraph:
        nodes = self.nodes  # pyright: ignore[reportUnknownVariableType]
        edges = self.edges  # pyright: ignore[reportUnknownVariableType]
        node_ids = [node.id for node in nodes]
        edge_ids = [edge.id for edge in edges]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node IDs must be unique")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Edge IDs must be unique")
        return self


###############################################################################
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    execution_backend: ExecutionBackend | None = None
    keras_enabled: bool = True
    graph: ModelGraph = Field(default_factory=ModelGraph)


###############################################################################
class ProjectUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    execution_backend: ExecutionBackend
    keras_enabled: bool = True
    graph: ModelGraph
    expected_revision: int = Field(ge=1)


###############################################################################
class ProjectDocument(BaseModel):
    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    schema_version: int
    execution_backend: ExecutionBackend
    keras_enabled: bool
    current_revision: int
    graph: ModelGraph
    validation_status: ValidationStatus
    generated_source: str | None = None


###############################################################################
class ProjectSummary(BaseModel):
    id: UUID
    name: str
    description: str
    updated_at: datetime
    execution_backend: ExecutionBackend
    keras_enabled: bool
    current_revision: int
    validation_status: ValidationStatus
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    has_generated_source: bool = False


###############################################################################
class ProjectListResponse(BaseModel):
    projects: list[ProjectSummary]


###############################################################################
class ProjectDuplicateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=160)


###############################################################################
class ProjectExportBundle(BaseModel):
    format: Literal["neuro-weave-project"] = "neuro-weave-project"
    format_version: Literal[5] = 5
    exported_at: datetime = Field(default_factory=utc_now)
    project: ProjectCreate


###############################################################################
class ShapeSummary(BaseModel):
    node_id: str
    outputs: dict[str, TensorSpec]
    parameter_count: int | None = None
