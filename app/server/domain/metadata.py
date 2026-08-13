from __future__ import annotations

from pydantic import BaseModel, Field

from domain.enums import CodeTarget, ExecutionBackend

###############################################################################
class HealthResponse(BaseModel):
    status: str

###############################################################################
class MetadataConfiguration(BaseModel):
    optimizers: list[str] = Field(default_factory=lambda: ["adam", "adamw", "sgd", "rmsprop"])
    losses: list[str] = Field(
        default_factory=lambda: [
            "categorical_crossentropy",
            "sparse_categorical_crossentropy",
            "binary_crossentropy",
            "mean_squared_error",
            "mean_absolute_error",
        ]
    )
    metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "auc", "mae", "mse"]
    )
    initializers: list[str] = Field(
        default_factory=lambda: ["glorot_uniform", "he_normal", "zeros", "ones"]
    )
    regularizers: list[str] = Field(default_factory=lambda: ["none", "l1", "l2", "l1_l2"])

###############################################################################
class MetadataResponse(BaseModel):
    name: str
    version: str
    schema_version: int
    code_targets: list[CodeTarget]
    execution_backends: list[ExecutionBackendStatus]
    configuration: MetadataConfiguration

###############################################################################
class ExecutionBackendStatus(BaseModel):
    id: ExecutionBackend
    available: bool
    reason_code: str | None = None
    accelerator: str | None = None
    device_name: str | None = None
