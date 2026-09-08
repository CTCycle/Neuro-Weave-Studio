from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, ShapeSummary

ModelTypeKind = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "embedding",
    "sequence",
    "multi_output",
    "generic",
]
InferenceConfidence = Literal["high", "medium", "low"]


###############################################################################
class ValidationDiagnostic(BaseModel):
    code: str
    severity: DiagnosticSeverity
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    parameter_id: str | None = None
    execution_backend: ExecutionBackend | None = None
    keras_enabled: bool | None = None
    details: dict[str, object] = Field(default_factory=dict)


###############################################################################
class OutputReference(BaseModel):
    node_id: str
    port_id: str
    name: str
    explicit: bool
    task_hint: str | None = None


###############################################################################
class ModelTypeInference(BaseModel):
    kind: ModelTypeKind
    confidence: InferenceConfidence
    evidence: list[str] = Field(default_factory=list)


###############################################################################
class ValidationResult(BaseModel):
    valid: bool
    diagnostics: list[ValidationDiagnostic]
    shape_summaries: list[ShapeSummary] = Field(default_factory=lambda: list[ShapeSummary]())
    total_parameters: int | None = None
    effective_outputs: list[OutputReference] = Field(default_factory=lambda: list[OutputReference]())
    output_mode: Literal["explicit", "inferred"] = "inferred"
    model_type: ModelTypeInference = Field(
        default_factory=lambda: ModelTypeInference(
            kind="generic", confidence="low", evidence=[]
        )
    )


###############################################################################
class ConnectionValidationRequest(BaseModel):
    graph: ModelGraph
    connection: EdgeInstance
    execution_backend: ExecutionBackend
    keras_enabled: bool = True


###############################################################################
class ConnectionValidationResult(BaseModel):
    valid: bool
    diagnostics: list[ValidationDiagnostic] = Field(default_factory=lambda: list[ValidationDiagnostic]())


###############################################################################
class ConnectionCandidatesValidationRequest(BaseModel):
    graph: ModelGraph
    connections: list[EdgeInstance] = Field(max_length=512)
    execution_backend: ExecutionBackend
    keras_enabled: bool = True


###############################################################################
class ConnectionCandidatesValidationResult(BaseModel):
    valid_connection_ids: list[str] = Field(default_factory=list)


###############################################################################
class ValidationRequest(BaseModel):
    graph: ModelGraph
    execution_backend: ExecutionBackend
    keras_enabled: bool = True
