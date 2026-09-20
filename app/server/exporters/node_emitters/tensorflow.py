from __future__ import annotations

from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    emit_multi_output_assignment,
    safe_index_expression,
)


###############################################################################
def _simple(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()), "")
    return EmissionResult(output_expressions=emit_multi_output_assignment(source, context.materialized_output_ports))


###############################################################################
def _constant(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = f"tf.broadcast_to(tf.cast(tf.convert_to_tensor({parameters.get('value', 0)!r}), {parameters.get('dtype', 'float32')!r}), {parameters.get('shape', [1])!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _parameter(context: NodeEmitterContext) -> EmissionResult:
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"self.{context.canonical_ir_node.variable_name}"})


###############################################################################
def _index(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    expression = safe_index_expression(context.normalized_parameters.get("expression", ":"))
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression.replace("x", source, 1)})


###############################################################################
def _average(context: NodeEmitterContext) -> EmissionResult:
    inputs = list(context.input_expressions_by_target_port.values())
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"tf.add_n([{', '.join(inputs)}]) / {len(inputs)}"})


###############################################################################
def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"tf.split({source}, num_or_size_splits={int(parameters.get('sections', 2))}, axis={int(parameters.get('axis', -1))})"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


###############################################################################
def _detection(context: NodeEmitterContext) -> EmissionResult:
    expression = "(" + ", ".join(context.input_expressions_by_target_port.values()) + ")"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


###############################################################################
def _reshape(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    target_shape = [int(value) for value in context.normalized_parameters.get("target_shape", [-1])]
    expression = f"tf.reshape({source}, [tf.shape({source})[0]] + {target_shape!r})"
    if context.normalized_parameters.get("dtype") is not None:
        expression = f"tf.cast({expression}, {context.normalized_parameters['dtype']!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _permute(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    axes = [int(value) for value in context.normalized_parameters.get("axes", [2, 1])]
    expression = f"tf.transpose({source}, [0] + {axes!r})"
    if context.normalized_parameters.get("dtype") is not None:
        expression = f"tf.cast({expression}, {context.normalized_parameters['dtype']!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
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
    if parameters.get("dtype") is not None:
        expression = f"tf.cast({expression}, {parameters['dtype']!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("tensorflow")
    authoritative_adapters = {
        "constant",
        "parameter",
        "index",
        "average",
        "split",
        "detection",
        "reshape",
        "permute",
        "repeat",
    }
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
        "attention": _simple,
        "feed_forward": _simple,
        "transformer_block": _simple,
        "positional_encoding": _simple,
        "residual": _simple,
        "safe_expression": _simple,
    }.items():
        registry.register(adapter, emitter, authoritative=adapter in authoritative_adapters)
    registry.declare_parameter_ids("constant", {"shape", "value", "dtype"})
    registry.declare_parameter_ids(
        "parameter", {"shape", "initializer", "dtype", "trainable"}
    )
    registry.declare_parameter_ids("average", set())
    registry.declare_parameter_ids("split", {"sections", "axis"})
    registry.declare_parameter_ids("detection", set())
    registry.declare_parameter_ids("index", {"expression"})
    registry.declare_parameter_ids("reshape", {"target_shape", "dtype"})
    registry.declare_parameter_ids("permute", {"axes", "dtype"})
    registry.declare_parameter_ids("repeat", {"repeats", "axis", "dtype"})
    return registry
