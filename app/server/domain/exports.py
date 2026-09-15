from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from domain.enums import ExecutionBackend
from domain.runtime import RuntimeApi


###############################################################################
class ExportFormat(StrEnum):
    PROJECT_BUNDLE = "project_bundle"
    SOURCE = "source"
    KERAS = "keras"
    KERAS_WEIGHTS = "keras_weights"
    TENSORFLOW_SAVED_MODEL = "tensorflow_saved_model"
    ONNX = "onnx"
    PYTORCH_EXPORTED_PROGRAM = "pytorch_exported_program"
    SAFETENSORS = "safetensors"
    TENSORFLOW_CHECKPOINT = "tensorflow_checkpoint"
    FLAX_CHECKPOINT = "flax_checkpoint"

###############################################################################
class ExportCategory(StrEnum):
    PROJECT = "project"
    SOURCE = "source"
    NATIVE = "native"
    INTERCHANGE = "interchange"
    CHECKPOINT = "checkpoint"

###############################################################################
class ExportLayout(StrEnum):
    FILE = "file"
    ARCHIVE = "archive"

###############################################################################
class ExportPayload(StrEnum):
    PROJECT = "project"
    SOURCE = "source"
    FULL_MODEL = "full_model"
    WEIGHTS_ONLY = "weights_only"

###############################################################################
class ExportConversion(StrEnum):
    CANONICAL = "canonical"
    NATIVE = "native"
    KERAS = "keras"
    BRIDGE = "bridge"

###############################################################################
class ExportVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structural: bool = False
    reload: bool = False
    inference: bool = False
    semantic_equivalence: Literal["verified", "not_claimed", "not_applicable"] = "not_claimed"
    notes: list[str] = Field(default_factory=list)

###############################################################################
class ExportArtifactFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    media_type: str = "application/octet-stream"

###############################################################################
class ExportArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: ExportFormat
    filename: str = Field(min_length=1)
    layout: ExportLayout
    payload: ExportPayload
    conversion: ExportConversion
    execution_backend: ExecutionBackend
    runtime_api: RuntimeApi | None = None
    project_id: UUID | None = None
    project_revision: int | None = Field(default=None, ge=1)
    graph_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    files: list[ExportArtifactFile] = Field(min_length=1)
    inputs: list[dict[str, Any]] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    outputs: list[dict[str, Any]] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    compilation: dict[str, Any] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    verification: ExportVerification = Field(default_factory=ExportVerification)

###############################################################################
class ExportOptionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    type: Literal["string", "integer", "boolean", "choice", "number"]
    default: Any = None
    choices: list[Any] = Field(default_factory=list)
    description: str = ""

###############################################################################
class ExportCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: ExportFormat
    label: str
    category: ExportCategory
    available: bool
    reason_code: str | None = None
    reason: str | None = None
    execution_backend: ExecutionBackend | None = None
    runtime_api: RuntimeApi | None = None
    conversion: ExportConversion
    layout: ExportLayout
    extension: str
    payload: ExportPayload
    inference_only: bool = True
    training_state: Literal["unavailable", "not_preserved"] = "unavailable"
    compatibility: str
    limitations: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    options: list[ExportOptionDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

###############################################################################
class ExportCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    project_revision: int
    execution_backend: ExecutionBackend
    keras_enabled: bool
    valid: bool
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    formats: list[ExportCapability]


###############################################################################
SUPPORTED_ONNX_OPSETS = (17, 18)

###############################################################################
class ExportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opset_version: int = Field(default=18, ge=SUPPORTED_ONNX_OPSETS[0], le=SUPPORTED_ONNX_OPSETS[-1])
    dynamic_batch: bool = True
    external_data: Literal["auto", "always", "never"] = "auto"
    max_shard_size_gb: float | None = Field(default=None, gt=0, le=100)

###############################################################################
def export_suffix(export_format: ExportFormat, options: ExportOptions | None = None) -> str:
    """Return the server-owned extension for a downloadable artifact."""

    suffixes = {
        ExportFormat.PROJECT_BUNDLE: ".neuro-weave.json",
        ExportFormat.SOURCE: ".py",
        ExportFormat.KERAS: ".keras",
        ExportFormat.KERAS_WEIGHTS: ".weights.h5",
        ExportFormat.TENSORFLOW_SAVED_MODEL: ".savedmodel.zip",
        ExportFormat.ONNX: ".onnx.zip"
        if options is not None and options.external_data == "always"
        else ".onnx",
        ExportFormat.PYTORCH_EXPORTED_PROGRAM: ".pt2",
        ExportFormat.SAFETENSORS: ".safetensors.zip",
        ExportFormat.TENSORFLOW_CHECKPOINT: ".tf-checkpoint.zip",
        ExportFormat.FLAX_CHECKPOINT: ".flax-checkpoint.zip",
    }
    return suffixes[export_format]

###############################################################################
class SourceExportResponse(BaseModel):
    execution_backend: ExecutionBackend
    keras_enabled: bool
    code: str


###############################################################################
