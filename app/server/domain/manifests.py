from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from domain.enums import CodeTarget, ExecutionBackend, PortKind

ParameterType = Literal[
    "string", "integer", "number", "boolean", "enum", "shape", "axis", "list", "tuple", "expression"
]

###############################################################################
class PortDefinition(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    display_name: str
    kind: PortKind = PortKind.TENSOR
    required: bool = True
    accepts_multiple: bool = False
    rank: int | None = Field(default=None, ge=0)
    element_kind: PortKind | None = None
    description: str = ""

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_element_kind(self) -> PortDefinition:
        if self.kind is PortKind.TENSOR_LIST and self.element_kind is None:
            return self.model_copy(update={"element_kind": PortKind.TENSOR})
        if self.kind is not PortKind.TENSOR_LIST and self.element_kind is not None:
            raise ValueError("Only tensor_list ports may declare element_kind")
        return self

###############################################################################
class ParameterDefinition(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    display_name: str
    type: ParameterType
    default: Any = None
    required: bool = False
    constraints: dict[str, Any] = Field(default_factory=dict)
    options: list[Any] = Field(default_factory=list)
    description: str = ""
    group: str = "General"
    nullable: bool = False
    ui: dict[str, Any] = Field(default_factory=dict)

###############################################################################
class ParameterCapability(BaseModel):
    """Declarative target-specific support for one canonical parameter."""

    supported: bool = True
    allowed_values: list[Any] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    mapping: str | None = None
    notes: str | None = None

###############################################################################
class FrameworkMapping(BaseModel):
    supported: bool = True
    symbol: str | None = None
    notes: str | None = None
    parameters: dict[str, ParameterCapability] = Field(default_factory=dict)
    execution_backends: dict[ExecutionBackend, bool] = Field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

###############################################################################
class UiMetadata(BaseModel):
    icon: str = "Box"
    accent: str = "slate"
    width: int = Field(default=250, ge=180, le=500)
    height: int = Field(default=120, ge=80, le=500)

###############################################################################
class ShapeRule(BaseModel):
    kind: str
    config: dict[str, Any] = Field(default_factory=dict)

###############################################################################
class NodeManifest(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    version: int = Field(default=1, ge=1)
    display_name: str
    category: str
    description: str
    inputs: list[PortDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    outputs: list[PortDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    parameters: list[ParameterDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    shape_rule: ShapeRule
    frameworks: dict[CodeTarget, FrameworkMapping]
    ui: UiMetadata = Field(default_factory=UiMetadata)
    deprecated: bool = False
    documentation_ref: str

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def unique_contract_names(self) -> NodeManifest:
        inputs = self.inputs  # pyright: ignore[reportUnknownVariableType]
        outputs = self.outputs  # pyright: ignore[reportUnknownVariableType]
        parameters = self.parameters  # pyright: ignore[reportUnknownVariableType]
        for label, definitions in (
            ("input", inputs),
            ("output", outputs),
            ("parameter", parameters),
        ):
            ids = [definition.id for definition in definitions]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} identifiers in manifest {self.id}")
        known_parameter_ids = {parameter.id for parameter in parameters}
        for target, mapping in self.frameworks.items():
            unknown_capabilities = sorted(set(mapping.parameters) - known_parameter_ids)
            if unknown_capabilities:
                raise ValueError(
                    f"Framework mapping {target.value} for {self.id} references unknown "
                    f"parameters: {', '.join(unknown_capabilities)}"
                )
        return self

###############################################################################
class NodeCatalogResponse(BaseModel):
    schema_version: int = 1
    nodes: list[NodeManifest]
