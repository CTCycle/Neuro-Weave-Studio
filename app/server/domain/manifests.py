from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from domain.enums import ExecutionBackend, PortKind

ParameterType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "enum",
    "shape",
    "axis",
    "list",
    "tuple",
    "object",
    "dtype",
    "activation",
    "initializer",
    "regularizer",
    "constraint",
    "expression",
]
NodeRole = Literal["input", "operation", "output_adapter"]
CapabilityStatus = Literal["verified", "unsupported", "unverified"]
OperationFamily = Literal["unary", "binary", "elementwise", "reduction", "tensor"]
OperationShapeBehavior = Literal[
    "preserve",
    "reduce",
    "reshape",
    "transpose",
    "expand_dims",
    "squeeze",
    "flatten",
    "repeat",
    "tile",
    "cast",
    "concatenate",
    "stack",
    "matmul",
    "take",
    "expression",
]
OperationOutputDtype = Literal["same", "bool", "int64", "parameter"]
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_OPERATION_SYMBOL_PREFIXES = {
    "keras": ("keras.ops.",),
    "tensorflow": ("tf.",),
    "pytorch": ("torch.",),
    "jax": ("jnp.", "jax.nn.", "jax.lax.", "jax.scipy.special."),
}

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
    enabled_when: dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_element_kind(self) -> PortDefinition:
        if self.kind is PortKind.TENSOR_LIST and self.element_kind is None:
            return self.model_copy(update={"element_kind": PortKind.TENSOR})
        if self.kind is not PortKind.TENSOR_LIST and self.element_kind is not None:
            raise ValueError("Only tensor_list ports may declare element_kind")
        return self

###############################################################################
class ParameterValueSchema(BaseModel):
    """Recursive JSON-safe value metadata shared by validation and the UI."""

    type: ParameterType
    item: ParameterValueSchema | None = None
    properties: dict[str, ParameterValueSchema] = Field(default_factory=dict)
    required_properties: list[str] = Field(default_factory=list)

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
    value_schema: ParameterValueSchema | None = None
    visible_when: dict[str, Any] = Field(default_factory=dict)
    enabled_when: dict[str, Any] = Field(default_factory=dict)
    required_when: dict[str, Any] = Field(default_factory=dict)
    framework_default: Any = None
    editor_default: Any = None

###############################################################################
class ParameterCapability(BaseModel):
    """Declarative target-specific support for one canonical parameter."""

    supported: bool = True
    status: CapabilityStatus = "verified"
    allowed_values: list[Any] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    mapping: str | None = None
    notes: str | None = None
    default: Any = None

###############################################################################
class OperationMetadata(BaseModel):
    """Typed metadata for a single-output mathematical/tensor operation."""

    family: OperationFamily
    shape_behavior: OperationShapeBehavior = "preserve"
    output_dtype: OperationOutputDtype = "same"
    arity: int = Field(default=1, ge=1, le=16)
    variadic: bool = False
    arguments: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_call_contract(self) -> OperationMetadata:
        if self.family == "unary" and self.arity != 1:
            raise ValueError("Unary operations must declare arity=1")
        if self.family == "reduction" and self.arity != 1:
            raise ValueError("Reduction operations must declare arity=1")
        if self.family in {"binary", "elementwise"} and self.arity != 2:
            raise ValueError("Binary and elementwise operations must declare arity=2")
        if self.variadic and self.family not in {"binary", "elementwise", "tensor"}:
            raise ValueError("Only binary, elementwise, and tensor operations may be variadic")
        parameter_ids = [*self.arguments, *self.keywords]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("Operation call parameters must be unique")
        return self

###############################################################################
class ImplementationMapping(BaseModel):
    """Capability metadata for one Keras or native node implementation."""

    supported: bool = True
    status: CapabilityStatus = "verified"
    symbol: str | None = None
    adapter: str | None = None
    notes: str | None = None
    parameters: dict[str, ParameterCapability] = Field(default_factory=dict)
    variant_symbols: dict[str, str] = Field(default_factory=dict)
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
    semantic_role: NodeRole = "operation"
    runtime_input: bool = True
    task_hint: str | None = None
    inputs: list[PortDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    outputs: list[PortDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    parameters: list[ParameterDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    shape_rule: ShapeRule
    implementations: dict[str, ImplementationMapping]
    operation: OperationMetadata | None = None
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
        if self.operation is not None:
            unknown_operation_parameters = sorted(
                set([*self.operation.arguments, *self.operation.keywords]) - known_parameter_ids
            )
            if unknown_operation_parameters:
                raise ValueError(
                    f"Operation {self.id} references unknown parameters: "
                    f"{', '.join(unknown_operation_parameters)}"
                )
        for implementation, mapping in self.implementations.items():
            unknown_capabilities = sorted(set(mapping.parameters) - known_parameter_ids)
            if unknown_capabilities:
                raise ValueError(
                    f"Implementation mapping {implementation} for {self.id} references unknown "
                    f"parameters: {', '.join(unknown_capabilities)}"
                )
            if mapping.symbol is not None and not _SYMBOL_PATTERN.fullmatch(mapping.symbol):
                raise ValueError(
                    f"Implementation mapping {implementation} for {self.id} has an invalid symbol"
                )
            for variant, symbol in mapping.variant_symbols.items():
                if not _SYMBOL_PATTERN.fullmatch(symbol):
                    raise ValueError(
                        f"Implementation mapping {implementation} for {self.id} has an invalid symbol for variant {variant}"
                    )
                if self.operation is not None:
                    prefixes = _OPERATION_SYMBOL_PREFIXES.get(implementation)
                    if prefixes and not symbol.startswith(prefixes):
                        raise ValueError(
                            f"Operation mapping {implementation} for {self.id} variant {variant} "
                            "must use a validated backend symbol"
                        )
            if (
                self.operation is not None
                and mapping.supported
                and mapping.symbol is None
                and mapping.adapter is None
            ):
                raise ValueError(
                    f"Supported operation mapping {implementation} for {self.id} must identify a symbol or adapter"
                )
            if self.operation is not None and mapping.supported and mapping.symbol is not None:
                prefixes = _OPERATION_SYMBOL_PREFIXES.get(implementation)
                if prefixes and not mapping.symbol.startswith(prefixes):
                    raise ValueError(
                        f"Operation mapping {implementation} for {self.id} must use a validated backend symbol"
                    )
        return self

###############################################################################
class NodeCatalogResponse(BaseModel):
    schema_version: int = 1
    nodes: list[NodeManifest]
