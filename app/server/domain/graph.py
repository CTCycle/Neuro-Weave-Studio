from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from common.timestamps import utc_now
from domain.compilation_plan import (
    CompilationObject,
    CompilationPlan,
    CompilerObject,
    normalize_compilation_plan,
)
from domain.enums import ExecutionBackend, ValidationStatus
from domain.tensor import TensorSpec

GRAPH_SCHEMA_VERSION = 6
PROJECT_BUNDLE_FORMAT_VERSION = 7


###############################################################################
def _mapping_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    keyword_values = getattr(value, "kwargs", None)
    return dict(keyword_values) if isinstance(keyword_values, dict) else {}

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

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        raw = _mapping_value(value)
        if "compilation" not in raw:
            raw["compilation"] = normalize_compilation_plan(raw)
        for key in (
            "optimizer",
            "optimizer_parameters",
            "losses",
            "metrics",
            "default_initializer",
            "default_regularizer",
            "default_constraint",
        ):
            raw.pop(key, None)
        return raw

    # Read-only compatibility accessors keep older callers source-compatible
    # while all serialized and validated state is the normalized plan.
    @property
    def optimizer(self) -> str:
        return self.compilation.optimizer.id

    @property
    def optimizer_parameters(self) -> dict[str, Any]:
        return self.compilation.optimizer.parameters

    @property
    def losses(self) -> dict[str, str]:
        return {key: value.id for key, value in self.compilation.losses.items()}

    @property
    def metrics(self) -> dict[str, list[str]]:
        return {key: [value.id for value in values] for key, values in self.compilation.metrics.items()}

    @property
    def default_initializer(self) -> str:
        return self.compilation.default_initializer.id

    @property
    def default_regularizer(self) -> str:
        return self.compilation.default_regularizer.id if self.compilation.default_regularizer else "none"

    @property
    def default_constraint(self) -> str:
        return self.compilation.default_constraint.id if self.compilation.default_constraint else "none"


# The former name remains an import alias, not an independent model.
KerasCompilationConfiguration = KerasProfileConfiguration

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

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        raw = _mapping_value(value)
        raw.setdefault("compilation", normalize_compilation_plan(raw))
        for key in ("optimizer", "optimizer_parameters", "losses", "metrics"):
            raw.pop(key, None)
        jit_compile = raw.pop("jit_compile", None)
        if "compiler" not in raw:
            raw["compiler"] = CompilerObject(
                id="tensorflow_function",
                parameters={"jit_compile": bool(jit_compile) if jit_compile is not None else False},
            )
        return raw

    @property
    def jit_compile(self) -> bool:
        return bool(self.compiler.parameters.get("jit_compile", False))

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

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        raw = _mapping_value(value)
        raw.setdefault("compilation", normalize_compilation_plan(raw))
        for key in ("optimizer", "optimizer_parameters", "losses", "metrics"):
            raw.pop(key, None)
        if "compiler" not in raw:
            raw["compiler"] = CompilerObject(
                id="pytorch_compile",
                parameters={
                    "enabled": bool(raw.pop("compile_enabled", False)),
                    "mode": raw.pop("mode", "default"),
                    "fullgraph": bool(raw.pop("fullgraph", False)),
                    "dynamic": bool(raw.pop("dynamic", False)),
                },
            )
        return raw

    @property
    def optimizer(self) -> str:
        return self.compilation.optimizer.id

    @property
    def optimizer_parameters(self) -> dict[str, Any]:
        return self.compilation.optimizer.parameters

    @property
    def losses(self) -> dict[str, str]:
        return {key: value.id for key, value in self.compilation.losses.items()}

    @property
    def metrics(self) -> dict[str, list[str]]:
        return {key: [value.id for value in values] for key, values in self.compilation.metrics.items()}

    @property
    def compile_enabled(self) -> bool:
        return bool(self.compiler.parameters.get("enabled", False))

    @property
    def mode(self) -> str:
        return str(self.compiler.parameters.get("mode", "default"))

    @property
    def fullgraph(self) -> bool:
        return bool(self.compiler.parameters.get("fullgraph", False))

    @property
    def dynamic(self) -> bool:
        return bool(self.compiler.parameters.get("dynamic", False))

###############################################################################
class JaxNativeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compilation: CompilationPlan = Field(default_factory=CompilationPlan)
    compiler: CompilerObject = Field(
        default_factory=lambda: CompilerObject(id="jax_jit", parameters={"enabled": False})
    )
    random_seed: int | None = 42

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        raw = _mapping_value(value)
        raw.setdefault("compilation", normalize_compilation_plan(raw))
        for key in ("optimizer", "optimizer_parameters", "losses", "metrics"):
            raw.pop(key, None)
        if "compiler" not in raw:
            raw["compiler"] = CompilerObject(
                id="jax_jit", parameters={"enabled": bool(raw.pop("jit_enabled", False))}
            )
        return raw

    @property
    def jit_enabled(self) -> bool:
        return bool(self.compiler.parameters.get("enabled", False))

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

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_configuration(cls, value: Any) -> Any:
        raw = _mapping_value(value)
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
    @model_validator(mode="before")
    @classmethod
    def normalize_import_payload(cls, value: Any) -> Any:
        from domain.serialization import normalize_graph_payload

        return normalize_graph_payload(value)

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

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def normalize_import_payload(cls, value: Any) -> Any:
        from domain.serialization import normalize_bundle_payload

        return normalize_bundle_payload(value)

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
