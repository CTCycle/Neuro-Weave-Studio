from __future__ import annotations

from pydantic import BaseModel, Field

from domain.diagnostics import ValidationDiagnostic
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph, OutputBinding, ShapeSummary

###############################################################################
class InputBinding(BaseModel):
    """A compiled edge retaining both endpoint port identities."""

    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str

###############################################################################
class CompiledNode(BaseModel):
    node_id: str
    definition_id: str
    variable_name: str
    input_node_ids: list[str]
    input_bindings: list[InputBinding]
    parameters: dict[str, object] = Field(default_factory=dict)

###############################################################################
class CanonicalModelIR(BaseModel):
    schema_version: int = 3
    execution_backend: ExecutionBackend
    keras_enabled: bool
    ordered_nodes: list[CompiledNode]
    input_node_ids: list[str]
    output_node_ids: list[str]
    output_bindings: list[OutputBinding] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    shape_summaries: list[ShapeSummary] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

###############################################################################
class CompilationResult(BaseModel):
    valid: bool
    execution_backend: ExecutionBackend
    keras_enabled: bool
    diagnostics: list[ValidationDiagnostic]
    code: str | None = None
    ir: CanonicalModelIR | None = None

###############################################################################
class CompilationRequest(BaseModel):
    graph: ModelGraph
    execution_backend: ExecutionBackend
    keras_enabled: bool = True
