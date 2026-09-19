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

###############################################################################
@keras.saving.register_keras_serializable(package="NeuroWeave")
class ConstantSource(layers.Layer):
    """Serializable constant graph source that is not a public model input."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        shape: list[int],
        value: Any,
        value_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source_shape = list(shape)
        self.source_value = value
        self.value_dtype = value_dtype

    # -------------------------------------------------------------------------
    def call(self, anchor: Any = None) -> Any:
        value = keras.ops.cast(
            keras.ops.convert_to_tensor(self.source_value), self.value_dtype
        )
        return keras.ops.broadcast_to(value, self.source_shape)

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "shape": self.source_shape,
            "value": self.source_value,
            "value_dtype": self.value_dtype,
        }

###############################################################################
@keras.saving.register_keras_serializable(package="NeuroWeave")
class ParameterSource(layers.Layer):
    """Serializable tracked source parameter with explicit trainability."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        shape: list[int],
        initializer: Any = "zeros",
        value_dtype: str = "float32",
        trainable: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(trainable=trainable, **kwargs)
        self.source_shape = list(shape)
        self.source_initializer = initializer
        self.value_dtype = value_dtype

    # -------------------------------------------------------------------------
    def build(self, input_shape: Any = None) -> None:
        del input_shape
        self.source_value = self.add_weight(
            name="value",
            shape=self.source_shape,
            initializer=keras.initializers.get(self.source_initializer),
            dtype=self.value_dtype,
            trainable=self.trainable,
        )

    # -------------------------------------------------------------------------
    def call(self, anchor: Any = None) -> Any:
        del anchor
        return self.source_value

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "shape": self.source_shape,
            "initializer": self.source_initializer,
            "value_dtype": self.value_dtype,
            "trainable": self.trainable,
        }
