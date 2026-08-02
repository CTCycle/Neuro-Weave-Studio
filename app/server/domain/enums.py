from __future__ import annotations

from enum import StrEnum


class TargetFramework(StrEnum):
    KERAS = "keras"
    TENSORFLOW_KERAS = "tensorflow_keras"
    PYTORCH = "pytorch"


class DiagnosticSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    COMPATIBILITY = "compatibility"
    INFO = "info"


class DataType(StrEnum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


class PortKind(StrEnum):
    TENSOR = "tensor"
    TENSOR_LIST = "tensor_list"
    SCALAR = "scalar"
    MASK = "mask"
