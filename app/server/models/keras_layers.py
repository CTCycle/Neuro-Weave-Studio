from __future__ import annotations

from typing import Any

import keras
from keras import layers


###############################################################################
@keras.saving.register_keras_serializable(package="NeuroWeave")
class SinusoidalPositionalEncoding(layers.Layer):
    """Backend-neutral sinusoidal positional encoding."""

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
        length = keras.ops.shape(inputs)[1]
        depth = keras.ops.shape(inputs)[2]
        positions = keras.ops.cast(keras.ops.arange(length)[:, None], inputs.dtype)
        dimensions = keras.ops.cast(keras.ops.arange(depth)[None, :], inputs.dtype)
        depth_value = keras.ops.cast(depth, inputs.dtype)
        rates = keras.ops.power(
            10000.0,
            -(2.0 * keras.ops.floor(dimensions / 2.0)) / depth_value,
        )
        angles = positions * rates
        even = keras.ops.cast(
            keras.ops.mod(keras.ops.arange(depth), 2) == 0,
            inputs.dtype,
        )[None, :]
        encoding = keras.ops.sin(angles) * even + keras.ops.cos(angles) * (1.0 - even)
        return inputs + encoding[None, :, :]

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return super().get_config()
