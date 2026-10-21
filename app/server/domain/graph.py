# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from common.timestamps import utc_now
from domain.compilation_plan import CompilationPlan, CompilerObject
from domain.enums import ExecutionBackend, ValidationStatus
from domain.tensor import TensorSpec

GRAPH_SCHEMA_VERSION = 6
PROJECT_BUNDLE_FORMAT_VERSION = 7

###############################################################################
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

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def normalize_parameters(self) -> NodeInstance:
        parameters = dict(self.parameters)
        if _empty_quantization_config(parameters.get("quantization_config")):
            parameters.pop("quantization_config", None)
        self.parameters = parameters
        return self

    # -------------------------------------------------------------------------
    @field_serializer("parameters")
    def serialize_parameters(self, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

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
class KerasProfileConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compilation: CompilationPlan = Field(default_factory=CompilationPlan)
    mixed_precision: bool = False
    random_seed: int | None = 42

###############################################################################
class TensorFlowNativeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compilation: CompilationPlan = Field(default_factory=CompilationPlan)
    compiler: CompilerObject = Field(
        default_factory=lambda: CompilerObject(
            id="tensorflow_function", parameters={"jit_compile": False}
        )
    )
    random_seed: int | None = 42

###############################################################################
class PyTorchNativeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compilation: CompilationPlan = Field(default_factory=CompilationPlan)
    compiler: CompilerObject = Field(
        default_factory=lambda: CompilerObject(
            id="pytorch_compile",
            parameters={"enabled": False, "mode": "default", "fullgraph": False, "dynamic": False},
        )
    )
    random_seed: int | None = 42

###############################################################################
class JaxNativeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compilation: CompilationPlan = Field(default_factory=CompilationPlan)
    compiler: CompilerObject = Field(
        default_factory=lambda: CompilerObject(id="jax_jit", parameters={"enabled": False})
    )
    random_seed: int | None = 42

###############################################################################
class NativeCompilationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensorflow: TensorFlowNativeConfiguration = Field(default_factory=TensorFlowNativeConfiguration)
    pytorch: PyTorchNativeConfiguration = Field(default_factory=PyTorchNativeConfiguration)
    jax: JaxNativeConfiguration = Field(default_factory=JaxNativeConfiguration)

###############################################################################
class ModelConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keras: KerasProfileConfiguration = Field(default_factory=KerasProfileConfiguration)
    native: NativeCompilationConfiguration = Field(default_factory=NativeCompilationConfiguration)

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_configuration(cls, value: Any) -> Any:
        raw = dict(cast(dict[str, Any], value)) if isinstance(value, dict) else {}
        raw.setdefault("keras", {})
        raw.setdefault("native", {})
        return raw

###############################################################################
class ExportSettings(BaseModel):
    include_comments: bool = True
    include_shape_assertions: bool = True
    preferred_binary_format: str = "keras"

###############################################################################
class ModelGraph(BaseModel):
    schema_version: Literal[6] = GRAPH_SCHEMA_VERSION
    nodes: list[NodeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    edges: list[EdgeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    output_bindings: list[OutputBinding] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
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
    execution_backend: ExecutionBackend
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
    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
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
    format_version: Literal[7] = PROJECT_BUNDLE_FORMAT_VERSION
    exported_at: datetime = Field(default_factory=utc_now)
    project: ProjectCreate

###############################################################################
class ShapeSummary(BaseModel):
    node_id: str
    outputs: dict[str, TensorSpec]
    parameter_count: int | None = None

###############################################################################
def _empty_quantization_config(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    if not value:
        return True
    mapping = cast(dict[str, Any], value)
    class_name = mapping.get("class_name")
    config = mapping.get("config")
    return (class_name is None or str(class_name).strip() == "") and (
        config is None or config == {} or config == []
    )
