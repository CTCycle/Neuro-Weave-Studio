from __future__ import annotations

from typing import Any, Literal, cast

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
        mapping = cast(dict[str, object], value)
        identifier = mapping.get("id", default_id)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Compilation objects require a non-empty id")
        raw_parameters = mapping.get("parameters", {})
        if not isinstance(raw_parameters, dict):
            raise ValueError("Compilation object parameters must be an object")
        parameters = cast(dict[str, Any], raw_parameters)
        return CompilationObject(id=identifier, parameters=parameters)
    if default_id is not None:
        return CompilationObject(id=default_id)
    raise ValueError("Compilation choices must be strings or objects")

###############################################################################
def normalize_compilation_plan(value: Any) -> CompilationPlan:
    """Normalize legacy string/id maps at the import boundary."""

    if isinstance(value, CompilationPlan):
        return value
    raw = dict(cast(dict[str, object], value)) if isinstance(value, dict) else {}
    optimizer_value = raw.get("optimizer")
    if optimizer_value is None:
        optimizer_value = raw.get("optimizer_id", "adam")
    optimizer_parameters = raw.get("optimizer_parameters", {})
    optimizer = normalize_compilation_object(optimizer_value, default_id="adam")
    if not optimizer.parameters and isinstance(optimizer_parameters, dict):
        optimizer = optimizer.model_copy(update={"parameters": optimizer_parameters})
    if not optimizer.parameters:
        default_parameters = {
            "adam": {"learning_rate": 0.001, "weight_decay": 0.0},
            "adamw": {"learning_rate": 0.001, "weight_decay": 0.004},
            "sgd": {
                "learning_rate": 0.01,
                "momentum": 0.0,
                "nesterov": False,
                "weight_decay": 0.0,
            },
            "rmsprop": {
                "learning_rate": 0.001,
                "rho": 0.9,
                "momentum": 0.0,
                "epsilon": 1e-7,
                "centered": False,
                "weight_decay": 0.0,
            },
            "adagrad": {
                "learning_rate": 0.001,
                "initial_accumulator_value": 0.1,
                "epsilon": 1e-7,
                "weight_decay": 0.0,
            },
            "adadelta": {
                "learning_rate": 0.001,
                "rho": 0.95,
                "epsilon": 1e-7,
                "weight_decay": 0.0,
            },
            "adamax": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-7,
                "weight_decay": 0.0,
            },
            "nadam": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-7,
                "weight_decay": 0.0,
            },
        }.get(optimizer.id)
        if default_parameters is not None:
            optimizer = optimizer.model_copy(update={"parameters": default_parameters})

    def normalize_losses(values: Any) -> dict[str, CompilationObject]:
        if not isinstance(values, dict):
            return {}
        typed_values = cast(dict[object, object], values)
        return {
            str(selector): normalize_compilation_object(choice)
            for selector, choice in typed_values.items()
        }

    def normalize_metrics(values: Any) -> dict[str, list[CompilationObject]]:
        if not isinstance(values, dict):
            return {}
        result: dict[str, list[CompilationObject]] = {}
        typed_values = cast(dict[object, Any], values)
        for selector, choices in typed_values.items():
            if isinstance(choices, list):
                typed_choices = cast(list[object], choices)
                result[str(selector)] = [
                    normalize_compilation_object(choice) for choice in typed_choices
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
        default_initializer=cast(
            CompilationObject,
            normalize_default(
                raw.get("default_initializer", "glorot_uniform"), "glorot_uniform", False
            ),
        ),
        default_regularizer=normalize_default(
            raw.get("default_regularizer", "none"), "none", True
        ),
        default_constraint=normalize_default(
            raw.get("default_constraint", "none"), "none", True
        ),
    )
