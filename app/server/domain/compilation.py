from __future__ import annotations

from pydantic import BaseModel, Field

from domain.diagnostics import ValidationDiagnostic
from domain.enums import TargetFramework


class CompiledNode(BaseModel):
    node_id: str
    definition_id: str
    variable_name: str
    input_node_ids: list[str]
    parameters: dict[str, object] = Field(default_factory=dict)


class CanonicalModelIR(BaseModel):
    schema_version: int = 1
    framework: TargetFramework
    ordered_nodes: list[CompiledNode]
    input_node_ids: list[str]
    output_node_ids: list[str]


class CompilationResult(BaseModel):
    valid: bool
    framework: TargetFramework
    diagnostics: list[ValidationDiagnostic]
    code: str | None = None
    ir: CanonicalModelIR | None = None
