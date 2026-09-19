from __future__ import annotations

from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    emit_multi_output_assignment,
)


def _simple(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()), "")
    return EmissionResult(output_expressions=emit_multi_output_assignment(source, context.materialized_output_ports))


def _constant(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = f"tf.broadcast_to(tf.cast(tf.convert_to_tensor({parameters.get('value', 0)!r}), {parameters.get('dtype', 'float32')!r}), {parameters.get('shape', [1])!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _parameter(context: NodeEmitterContext) -> EmissionResult:
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"self.{context.canonical_ir_node.variable_name}"})


def _index(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    value = context.normalized_parameters.get("expression", ":")
    expression = value if isinstance(value, str) and value.startswith("x") else f"x[{value}]"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression.replace("x", source, 1)})


def _average(context: NodeEmitterContext) -> EmissionResult:
    inputs = list(context.input_expressions_by_target_port.values())
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"tf.add_n([{', '.join(inputs)}]) / {len(inputs)}"})


def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"tf.split({source}, num_or_size_splits={int(parameters.get('sections', 2))}, axis={int(parameters.get('axis', -1))})"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _detection(context: NodeEmitterContext) -> EmissionResult:
    expression = "(" + ", ".join(context.input_expressions_by_target_port.values()) + ")"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _reshape(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    target_shape = [int(value) for value in context.normalized_parameters.get("target_shape", [-1])]
    expression = f"tf.reshape({source}, [tf.shape({source})[0]] + {target_shape!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _permute(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    axes = [int(value) for value in context.normalized_parameters.get("axes", [2, 1])]
    expression = f"tf.transpose({source}, [0] + {axes!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _repeat(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    repeats = int(parameters.get("repeats", 2))
    axis = int(parameters.get("axis", 1))
    expression = (
        f"tf.repeat(tf.expand_dims({source}, axis=1), {repeats}, axis=1)"
        if axis == 1
        else f"tf.repeat({source}, {repeats}, axis={axis})"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("tensorflow")
    for adapter, emitter in {
        "identity": _simple,
        "constant": _constant,
        "parameter": _parameter,
        "index": _index,
        "average": _average,
        "split": _split,
        "detection": _detection,
        "reshape": _reshape,
        "permute": _permute,
        "repeat": _repeat,
        "group_norm": _simple,
        "prelu": _simple,
        "rms_norm": _simple,
        "unit_norm": _simple,
        "pool_nd": _simple,
        "global_pool_nd": _simple,
        "upsampling": _simple,
        "padding": _simple,
        "cropping": _simple,
        "conv3d": _simple,
        "depthwise": _simple,
        "separable": _simple,
        "transposed": _simple,
        "simple_rnn": _simple,
        "bidirectional": _simple,
        "stack": _simple,
        "positional_encoding": _simple,
        "residual": _simple,
        "safe_expression": _simple,
    }.items():
        registry.register(adapter, emitter)
    return registry
