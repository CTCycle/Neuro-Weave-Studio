from __future__ import annotations

from pydantic import BaseModel, Field

from domain.enums import DiagnosticSeverity, TargetFramework
from domain.graph import ModelGraph, ShapeSummary


###############################################################################
class ValidationDiagnostic(BaseModel):
    code: str
    severity: DiagnosticSeverity
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    framework: TargetFramework | None = None
    details: dict[str, object] = Field(default_factory=dict)


###############################################################################
class ValidationResult(BaseModel):
    valid: bool
    diagnostics: list[ValidationDiagnostic]
    shape_summaries: list[ShapeSummary] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    total_parameters: int | None = None


###############################################################################
class ValidationRequest(BaseModel):
    graph: ModelGraph
    framework: TargetFramework
