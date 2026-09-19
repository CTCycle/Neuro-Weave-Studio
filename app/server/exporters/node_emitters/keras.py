from __future__ import annotations

from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    emit_multi_output_assignment,
)


def _identity(context: NodeEmitterContext) -> EmissionResult:
    expression = next(iter(context.input_expressions_by_target_port.values()), "")
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _constant(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = (
        f"keras.ops.broadcast_to(keras.ops.cast(keras.ops.convert_to_tensor({parameters.get('value', 0)!r}), "
        f"{parameters.get('dtype', 'float32')!r}), {parameters.get('shape', [1])!r})"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _parameter(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = (
        f"ParameterSource(shape={parameters.get('shape', [1])!r}, "
        f"initializer={parameters.get('initializer', 'zeros')!r}, "
        f"value_dtype={parameters.get('dtype', 'float32')!r}, "
        f"trainable={parameters.get('trainable', True)!r})()"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _index(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    value = context.normalized_parameters.get("expression", ":")
    expression = value if isinstance(value, str) and value.startswith("x") else f"x[{value}]"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression.replace("x", source, 1)})


def _average(context: NodeEmitterContext) -> EmissionResult:
    inputs = list(context.input_expressions_by_target_port.values())
    expression = f"keras.ops.sum(keras.ops.stack([{', '.join(inputs)}], axis=0), axis=0) / {len(inputs)}"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"keras.ops.split({source}, {int(parameters.get('sections', 2))}, axis={int(parameters.get('axis', -1))})"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _detection(context: NodeEmitterContext) -> EmissionResult:
    expression = "[" + ", ".join(context.input_expressions_by_target_port.values()) + "]"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _reshape(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    target_shape = tuple(int(value) for value in parameters.get("target_shape", [-1]))
    expression = f"keras.layers.Reshape({target_shape!r}, name={context.canonical_ir_node.variable_name!r})({source})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _permute(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    axes = tuple(int(value) for value in parameters.get("axes", [2, 1]))
    expression = f"keras.layers.Permute({axes!r}, name={context.canonical_ir_node.variable_name!r})({source})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _repeat(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    repeats = int(parameters.get("repeats", 2))
    axis = int(parameters.get("axis", 1))
    if axis == 1:
        expression = f"keras.layers.RepeatVector({repeats}, name={context.canonical_ir_node.variable_name!r})({source})"
    else:
        expression = f"keras.ops.repeat({source}, {repeats}, axis={axis})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("keras")
    for adapter, emitter in {
        "identity": _identity,
        "constant": _constant,
        "parameter": _parameter,
        "index": _index,
        "average": _average,
        "split": _split,
        "detection": _detection,
        "keras_reshape": _reshape,
        "keras_permute": _permute,
        "keras_repeat": _repeat,
        "keras_group_norm": _identity,
        "prelu": _identity,
        "rms_norm": _identity,
        "unit_norm": _identity,
        "keras_max_pool1d": _identity,
        "keras_max_pool2d": _identity,
        "keras_max_pool3d": _identity,
        "keras_average_pool1d": _identity,
        "keras_average_pool2d": _identity,
        "keras_average_pool3d": _identity,
        "keras_global_pool1d": _identity,
        "keras_global_pool2d": _identity,
        "keras_global_pool3d": _identity,
        "keras_upsampling": _identity,
        "keras_padding": _identity,
        "keras_cropping": _identity,
        "keras_conv3d": _identity,
        "keras_depthwise": _identity,
        "keras_separable": _identity,
        "keras_transposed": _identity,
        "safe_expression": _identity,
    }.items():
        registry.register(adapter, emitter)
    return registry
