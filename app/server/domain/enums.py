# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from enum import StrEnum


###############################################################################
class ExecutionBackend(StrEnum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    JAX = "jax"

###############################################################################
def keras_backend_value(backend: ExecutionBackend) -> str:
    """Map the public PyTorch id to the Keras worker value ``torch``."""

    return "torch" if backend is ExecutionBackend.PYTORCH else backend.value

###############################################################################
class ValidationStatus(StrEnum):
    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"

###############################################################################
class DiagnosticSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    COMPATIBILITY = "compatibility"
    INFO = "info"

###############################################################################
class DataType(StrEnum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"

###############################################################################
class PortKind(StrEnum):
    TENSOR = "tensor"
    TENSOR_LIST = "tensor_list"
    SCALAR = "scalar"
    MASK = "mask"
