from __future__ import annotations

from pydantic import BaseModel, Field

from domain.enums import CodeTarget, DiagnosticSeverity
from domain.graph import ModelGraph, ShapeSummary


class ValidationDiagnostic(BaseModel):
    code: str
    severity: DiagnosticSeverity
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    code_target: CodeTarget | None = None
    details: dict[str, object] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    valid: bool
    diagnostics: list[ValidationDiagnostic]
    shape_summaries: list[ShapeSummary] = Field(default_factory=lambda: list[ShapeSummary]())
    total_parameters: int | None = None


class ValidationRequest(BaseModel):
    graph: ModelGraph
    code_target: CodeTarget
