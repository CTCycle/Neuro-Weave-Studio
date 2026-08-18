from __future__ import annotations

from pydantic import BaseModel, Field

from domain.diagnostics import ValidationDiagnostic
from domain.enums import CodeTarget
from domain.graph import ModelGraph, ShapeSummary

###############################################################################
class CompiledNode(BaseModel):
    node_id: str
    definition_id: str
    variable_name: str
    input_node_ids: list[str]
    parameters: dict[str, object] = Field(default_factory=dict)

###############################################################################
class CanonicalModelIR(BaseModel):
    schema_version: int = 2
    code_target: CodeTarget
    ordered_nodes: list[CompiledNode]
    input_node_ids: list[str]
    output_node_ids: list[str]
    shape_summaries: list[ShapeSummary] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

###############################################################################
class CompilationResult(BaseModel):
    valid: bool
    code_target: CodeTarget
    diagnostics: list[ValidationDiagnostic]
    code: str | None = None
    ir: CanonicalModelIR | None = None

###############################################################################
class CompilationRequest(BaseModel):
    graph: ModelGraph
    code_target: CodeTarget
