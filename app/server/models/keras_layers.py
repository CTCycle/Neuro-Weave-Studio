# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from typing import Any

import keras
from keras import layers


###############################################################################
@keras.saving.register_keras_serializable(package="NeuroWeave")
class SinusoidalPositionalEncoding(layers.Layer):
    """Backend-neutral sinusoidal positional encoding."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        sequence_length: int | None = None,
        feature_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
        input_shape = getattr(inputs, "shape", None)
        static_length = (
            input_shape[1]
            if input_shape is not None and len(input_shape) > 1
            else None
        )
        static_depth = (
            input_shape[2]
            if input_shape is not None and len(input_shape) > 2
            else None
        )
        length = self.sequence_length or static_length or keras.ops.shape(inputs)[1]
        depth = self.feature_dim or static_depth or keras.ops.shape(inputs)[2]
        positions = keras.ops.cast(keras.ops.arange(0, length, 1)[:, None], inputs.dtype)
        dimensions = keras.ops.cast(keras.ops.arange(0, depth, 1)[None, :], inputs.dtype)
        depth_value = keras.ops.cast(depth, inputs.dtype)
        rates = keras.ops.power(
            10000.0,
            -(2.0 * keras.ops.floor(dimensions / 2.0)) / depth_value,
        )
        angles = positions * rates
        even = keras.ops.cast(
            keras.ops.mod(keras.ops.arange(0, depth, 1), 2) == 0,
            inputs.dtype,
        )[None, :]
        encoding = keras.ops.sin(angles) * even + keras.ops.cos(angles) * (1.0 - even)
        return inputs + encoding[None, :, :]

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
        }

###############################################################################
@keras.saving.register_keras_serializable(package="NeuroWeave")
class GeluActivation(layers.Layer):
    """Backend-neutral exact GELU for export bridges without a GELU primitive."""

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
        half = keras.ops.cast(0.5, inputs.dtype)
        one = keras.ops.cast(1.0, inputs.dtype)
        root_two = keras.ops.cast(2.0, inputs.dtype)
        return half * inputs * (one + keras.ops.erf(inputs / keras.ops.sqrt(root_two)))

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return super().get_config()
