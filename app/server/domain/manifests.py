from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from domain.enums import PortKind, TargetFramework

ParameterType = Literal[
    "string", "integer", "number", "boolean", "enum", "shape", "axis", "expression"
]


class PortDefinition(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    display_name: str
    kind: PortKind = PortKind.TENSOR
    required: bool = True
    accepts_multiple: bool = False
    rank: int | None = Field(default=None, ge=0)
    description: str = ""


class ParameterDefinition(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    display_name: str
    type: ParameterType
    default: Any = None
    required: bool = False
    constraints: dict[str, Any] = Field(default_factory=dict)
    options: list[Any] = Field(default_factory=list)
    description: str = ""


class FrameworkMapping(BaseModel):
    supported: bool = True
    symbol: str | None = None
    notes: str | None = None


class UiMetadata(BaseModel):
    icon: str = "Box"
    accent: str = "slate"
    width: int = Field(default=250, ge=180, le=500)
    height: int = Field(default=120, ge=80, le=500)


class ShapeRule(BaseModel):
    kind: str
    config: dict[str, Any] = Field(default_factory=dict)


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
    frameworks: dict[TargetFramework, FrameworkMapping]
    ui: UiMetadata = Field(default_factory=UiMetadata)
    deprecated: bool = False
    documentation_ref: str

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
        return self


class NodeCatalogResponse(BaseModel):
    schema_version: int = 1
    nodes: list[NodeManifest]
