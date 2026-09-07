from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from common.timestamps import utc_now
from domain.enums import ExecutionBackend, ValidationStatus
from domain.tensor import TensorSpec

GRAPH_SCHEMA_VERSION = 5
PROJECT_BUNDLE_FORMAT_VERSION = 6

###############################################################################
class Position(BaseModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)

###############################################################################
class RuntimeInputParameters(dict[str, Any]):
    """Persist feature-only input shapes while exposing a runtime batch axis."""

    # -------------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return _runtime_input_value(key, super().__getitem__(key))

    # -------------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        if key not in self:
            return default
        return _runtime_input_value(key, super().get(key))

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
    def normalize_runtime_parameters(self) -> NodeInstance:
        parameters = dict(self.parameters)
        if _empty_quantization_config(parameters.get("quantization_config")):
            parameters.pop("quantization_config", None)
        self.parameters = (
            RuntimeInputParameters(parameters)
            if self.definition_id.startswith("input.")
            else parameters
        )
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
    schema_version: Literal[5] = GRAPH_SCHEMA_VERSION
    nodes: list[NodeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    edges: list[EdgeInstance] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    output_bindings: list[OutputBinding] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    model_configuration: ModelConfiguration = Field(default_factory=ModelConfiguration)
    export_settings: ExportSettings = Field(default_factory=ExportSettings)
    future_training_configuration: dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_graph(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw_value = cast(dict[str, Any], value)
        if raw_value.get("schema_version") != 4:
            return raw_value
        migrated: dict[str, Any] = dict(raw_value)
        migrated["schema_version"] = GRAPH_SCHEMA_VERSION
        nodes: list[Any] = []
        for raw_node in cast(list[Any], raw_value.get("nodes", [])):
            if isinstance(raw_node, NodeInstance):
                node: dict[str, Any] = raw_node.model_dump(mode="python")
            elif isinstance(raw_node, dict):
                node = dict(cast(dict[str, Any], raw_node))
            else:
                nodes.append(raw_node)
                continue
            parameters = dict(cast(dict[str, Any], node.get("parameters", {})))
            if str(node.get("definition_id", "")).startswith("input."):
                shape = parameters.get("shape")
                if isinstance(shape, list) and shape and shape[0] is None:
                    parameters["shape"] = shape[1:]
            if _empty_quantization_config(parameters.get("quantization_config")):
                parameters.pop("quantization_config", None)
            node["parameters"] = parameters
            nodes.append(node)
        migrated["nodes"] = nodes
        return migrated

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
    format_version: Literal[6] = PROJECT_BUNDLE_FORMAT_VERSION
    exported_at: datetime = Field(default_factory=utc_now)
    project: ProjectCreate

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_bundle(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw_value = cast(dict[str, Any], value)
        if raw_value.get("format_version") != 5:
            return raw_value
        migrated: dict[str, Any] = dict(raw_value)
        migrated["format_version"] = PROJECT_BUNDLE_FORMAT_VERSION
        return migrated

###############################################################################
class ShapeSummary(BaseModel):
    node_id: str
    outputs: dict[str, TensorSpec]
    parameter_count: int | None = None


###############################################################################
def _runtime_input_value(key: str, value: Any) -> Any:
    if key == "shape" and isinstance(value, list):
        return [None, *value]
    return value


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
