# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

# Keras currently ships no usable typing stubs for this installed package.
# Keep its dynamic API behind an explicit framework boundary.
keras_api: Any = cast(Any, import_module("keras"))
keras_layers: Any = keras_api.layers

###############################################################################
@keras_api.saving.register_keras_serializable(package="NeuroWeave")
class SinusoidalPositionalEncoding(keras_layers.Layer):
    """Backend-neutral sinusoidal positional encoding."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        sequence_length: int | None = None,
        feature_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        base: Any = super()
        base.__init__(**kwargs)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
        input_shape = getattr(inputs, "shape", None)
        static_length = _static_dimension(input_shape, 1)
        static_depth = _static_dimension(input_shape, 2)
        dynamic_shape = keras_api.ops.shape(inputs)
        length: Any = self.sequence_length or static_length or dynamic_shape[1]
        depth: Any = self.feature_dim or static_depth or dynamic_shape[2]
        position_range: Any = keras_api.ops.arange(0, length, 1)
        dimension_range: Any = keras_api.ops.arange(0, depth, 1)
        positions: Any = keras_api.ops.cast(position_range[:, None], inputs.dtype)
        dimensions: Any = keras_api.ops.cast(dimension_range[None, :], inputs.dtype)
        depth_value: Any = keras_api.ops.cast(depth, inputs.dtype)
        scaled_dimensions: Any = dimensions / 2.0
        floored_dimensions: Any = keras_api.ops.floor(scaled_dimensions)
        rate_exponents: Any = -(2.0 * floored_dimensions) / depth_value
        rates: Any = keras_api.ops.power(
            10000.0,
            rate_exponents,
        )
        angles: Any = positions * rates
        parity: Any = keras_api.ops.mod(dimension_range, 2)
        even_values: Any = keras_api.ops.cast(parity == 0, inputs.dtype)
        even: Any = even_values[None, :]
        encoding: Any = keras_api.ops.sin(angles) * even + keras_api.ops.cos(angles) * (1.0 - even)
        return inputs + encoding[None, :, :]

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base: Any = super()
        return {
            **base.get_config(),
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
        }

###############################################################################
def _static_dimension(shape: Any, axis: int) -> Any:
    """Read one optional static dimension from a backend tensor shape."""
    if shape is None or len(shape) <= axis:
        return None
    return shape[axis]

###############################################################################
@keras_api.saving.register_keras_serializable(package="NeuroWeave")
class GeluActivation(keras_layers.Layer):
    """Backend-neutral exact GELU for export bridges without a GELU primitive."""

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
        half: Any = keras_api.ops.cast(0.5, inputs.dtype)
        one: Any = keras_api.ops.cast(1.0, inputs.dtype)
        root_two: Any = keras_api.ops.cast(2.0, inputs.dtype)
        return half * inputs * (one + keras_api.ops.erf(inputs / keras_api.ops.sqrt(root_two)))

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base: Any = super()
        return base.get_config()

###############################################################################
@keras_api.saving.register_keras_serializable(package="NeuroWeave")
class ConstantSource(keras_layers.Layer):
    """Serializable constant graph source that is not a public model input."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        shape: list[int],
        value: Any,
        value_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        base: Any = super()
        base.__init__(**kwargs)
        self.source_shape = list(shape)
        self.source_value = value
        self.value_dtype = value_dtype

    # -------------------------------------------------------------------------
    def call(self, anchor: Any = None) -> Any:
        value = keras_api.ops.cast(
            keras_api.ops.convert_to_tensor(self.source_value), self.value_dtype
        )
        return keras_api.ops.broadcast_to(value, self.source_shape)

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base: Any = super()
        return {
            **base.get_config(),
            "shape": self.source_shape,
            "value": self.source_value,
            "value_dtype": self.value_dtype,
        }

###############################################################################
@keras_api.saving.register_keras_serializable(package="NeuroWeave")
class ParameterSource(keras_layers.Layer):
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
        base: Any = super()
        base.__init__(trainable=trainable, **kwargs)
        self.source_shape = list(shape)
        self.source_initializer = initializer
        self.value_dtype = value_dtype

    # -------------------------------------------------------------------------
    def build(self, input_shape: Any = None) -> None:
        del input_shape
        self.source_value = self.add_weight(
            name="value",
            shape=self.source_shape,
            initializer=keras_api.initializers.get(self.source_initializer),
            dtype=self.value_dtype,
            trainable=self.trainable,
        )

    # -------------------------------------------------------------------------
    def call(self, anchor: Any = None) -> Any:
        del anchor
        return self.source_value

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base: Any = super()
        return {
            **base.get_config(),
            "shape": self.source_shape,
            "initializer": self.source_initializer,
            "value_dtype": self.value_dtype,
            "trainable": self.trainable,
        }

###############################################################################
@keras_api.saving.register_keras_serializable(package="NeuroWeave")
class StaticGraphModel(keras_api.Model):
    """Serializable wrapper for graphs with no public runtime inputs."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        inner_model: Any,
        anchor_shape: tuple[int, ...] = (1,),
        **kwargs: Any,
    ) -> None:
        base: Any = super()
        base.__init__(**kwargs)
        self.inner_model = inner_model
        self.anchor_shape = tuple(anchor_shape)

    # -------------------------------------------------------------------------
    def call(self, inputs: Any = None, training: bool = False) -> Any:
        del inputs
        anchor = keras_api.ops.zeros((1, *self.anchor_shape), dtype="float32")
        return self.inner_model(anchor, training=training)

    # -------------------------------------------------------------------------
    def get_build_config(self) -> dict[str, Any]:
        return {}

    # -------------------------------------------------------------------------
    def build_from_config(self, config: dict[str, Any]) -> None:
        del config
        self.built = True

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base: Any = super()
        return {
            **base.get_config(),
            "inner_model": keras_api.saving.serialize_keras_object(self.inner_model),
            "anchor_shape": list(self.anchor_shape),
        }

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        custom_objects: dict[str, Any] | None = None,
    ) -> StaticGraphModel:
        inner_model = keras_api.saving.deserialize_keras_object(
            config.pop("inner_model"), custom_objects=custom_objects
        )
        return cls(inner_model=inner_model, **config)
