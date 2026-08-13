from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from common.timestamps import utc_now
from domain.enums import CodeTarget, ValidationStatus
from domain.tensor import TensorSpec

GRAPH_SCHEMA_VERSION = 2

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

###############################################################################
class EdgeInstance(BaseModel):
    id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str

###############################################################################
class ModelConfiguration(BaseModel):
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
class ExportSettings(BaseModel):
    include_comments: bool = True
    include_shape_assertions: bool = True
    preferred_binary_format: str = "keras"

###############################################################################
class ModelGraph(BaseModel):
    schema_version: Literal[2] = GRAPH_SCHEMA_VERSION
    nodes: list[NodeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    edges: list[EdgeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
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
    code_target: CodeTarget = CodeTarget.KERAS
    graph: ModelGraph = Field(default_factory=ModelGraph)

###############################################################################
class ProjectUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    code_target: CodeTarget
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
    code_target: CodeTarget
    current_revision: int
    graph: ModelGraph
    validation_status: ValidationStatus
    generated_code_snapshots: dict[str, str] = Field(default_factory=dict)

###############################################################################
def _empty_code_targets() -> list[CodeTarget]:
    return []

###############################################################################
class ProjectSummary(BaseModel):
    id: UUID
    name: str
    description: str
    updated_at: datetime
    code_target: CodeTarget
    current_revision: int
    validation_status: ValidationStatus
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    compiled_targets: list[CodeTarget] = Field(default_factory=_empty_code_targets)

###############################################################################
class ProjectListResponse(BaseModel):
    projects: list[ProjectSummary]

###############################################################################
class ProjectDuplicateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=160)

###############################################################################
class ProjectExportBundle(BaseModel):
    format: Literal["neuro-weave-project"] = "neuro-weave-project"
    format_version: Literal[2] = 2
    exported_at: datetime = Field(default_factory=utc_now)
    project: ProjectCreate

###############################################################################
class ShapeSummary(BaseModel):
    node_id: str
    outputs: dict[str, TensorSpec]
    parameter_count: int | None = None
