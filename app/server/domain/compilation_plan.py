from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CompilerId = Literal["tensorflow_function", "pytorch_compile", "jax_jit"]


###############################################################################
class CompilationObject(BaseModel):
    """A catalog-backed executable choice with normalized parameters."""

    id: str = Field(min_length=1, max_length=160)
    parameters: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class CompilationPlan(BaseModel):
    """Shared compilation representation for Keras and native profiles."""

    optimizer: CompilationObject = Field(
        default_factory=lambda: CompilationObject(
            id="adam", parameters={"learning_rate": 0.001}
        )
    )
    losses: dict[str, CompilationObject] = Field(default_factory=dict)
    metrics: dict[str, list[CompilationObject]] = Field(default_factory=dict)
    default_initializer: CompilationObject = Field(
        default_factory=lambda: CompilationObject(id="glorot_uniform")
    )
    default_regularizer: CompilationObject | None = None
    default_constraint: CompilationObject | None = None


###############################################################################
class CompilerObject(BaseModel):
    id: CompilerId
    parameters: dict[str, Any] = Field(default_factory=dict)


###############################################################################
def normalize_compilation_object(
    value: Any, *, default_id: str | None = None
) -> CompilationObject:
    if isinstance(value, CompilationObject):
        return value
    if isinstance(value, str):
        return CompilationObject(id=value)
    if isinstance(value, dict):
        identifier = value.get("id", default_id)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Compilation objects require a non-empty id")
        parameters = value.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Compilation object parameters must be an object")
        return CompilationObject(id=identifier, parameters=parameters)
    if default_id is not None:
        return CompilationObject(id=default_id)
    raise ValueError("Compilation choices must be strings or objects")


###############################################################################
def normalize_compilation_plan(value: Any) -> CompilationPlan:
    """Normalize legacy string/id maps at the import boundary."""

    if isinstance(value, CompilationPlan):
        return value
    raw = dict(value) if isinstance(value, dict) else {}
    optimizer_value = raw.get("optimizer")
    if optimizer_value is None:
        optimizer_value = raw.get("optimizer_id", "adam")
    optimizer_parameters = raw.get("optimizer_parameters", {})
    optimizer = normalize_compilation_object(optimizer_value, default_id="adam")
    if not optimizer.parameters and isinstance(optimizer_parameters, dict):
        optimizer = optimizer.model_copy(update={"parameters": optimizer_parameters})

    def normalize_losses(values: Any) -> dict[str, CompilationObject]:
        if not isinstance(values, dict):
            return {}
        return {
            str(selector): normalize_compilation_object(choice)
            for selector, choice in values.items()
        }

    def normalize_metrics(values: Any) -> dict[str, list[CompilationObject]]:
        if not isinstance(values, dict):
            return {}
        result: dict[str, list[CompilationObject]] = {}
        for selector, choices in values.items():
            if isinstance(choices, list):
                result[str(selector)] = [
                    normalize_compilation_object(choice) for choice in choices
                ]
            elif choices is not None:
                result[str(selector)] = [normalize_compilation_object(choices)]
        return result

    def normalize_default(
        value: Any, fallback: str, nullable: bool
    ) -> CompilationObject | None:
        if value in (None, "none") and nullable:
            return None
        return normalize_compilation_object(value, default_id=fallback)

    return CompilationPlan(
        optimizer=optimizer,
        losses=normalize_losses(raw.get("losses", raw.get("loss_ids", {}))),
        metrics=normalize_metrics(raw.get("metrics", raw.get("metric_ids", {}))),
        default_initializer=normalize_default(
            raw.get("default_initializer", "glorot_uniform"), "glorot_uniform", False
        ),
        default_regularizer=normalize_default(
            raw.get("default_regularizer", "none"), "none", True
        ),
        default_constraint=normalize_default(
            raw.get("default_constraint", "none"), "none", True
        ),
    )

