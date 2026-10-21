# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

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
