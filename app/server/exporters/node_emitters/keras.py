# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    NodeRuntimeContext,
    emit_multi_output_assignment,
    safe_index_expression,
)


###############################################################################
def _identity(context: NodeEmitterContext) -> EmissionResult:
    expression = next(iter(context.input_expressions_by_target_port.values()), "")
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))

###############################################################################
def _constant(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    anchor = f"({context.anchor_expression})" if context.anchor_expression else "()"
    expression = (
        f"ConstantSource(shape={parameters.get('shape', [1])!r}, "
        f"value={parameters.get('value', 0)!r}, "
        f"value_dtype={parameters.get('dtype', 'float32')!r}, "
        f"name={context.canonical_ir_node.variable_name!r}){anchor}"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _parameter(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    anchor = f"({context.anchor_expression})" if context.anchor_expression else "()"
    expression = (
        f"ParameterSource(shape={parameters.get('shape', [1])!r}, "
        f"initializer={parameters.get('initializer', 'zeros')!r}, "
        f"value_dtype={parameters.get('dtype', 'float32')!r}, "
        f"trainable={parameters.get('trainable', True)!r}, "
        f"name={context.canonical_ir_node.variable_name!r}){anchor}"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _index(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    expression = safe_index_expression(context.normalized_parameters.get("expression", ":"))
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression.replace("x", source, 1)})

###############################################################################
def _average(context: NodeEmitterContext) -> EmissionResult:
    inputs = list(context.input_expressions_by_target_port.values())
    expression = f"keras.ops.sum(keras.ops.stack([{', '.join(inputs)}], axis=0), axis=0) / {len(inputs)}"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"keras.ops.split({source}, {int(parameters.get('sections', 2))}, axis={int(parameters.get('axis', -1))})"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))

###############################################################################
def _detection(context: NodeEmitterContext) -> EmissionResult:
    expression = "[" + ", ".join(context.input_expressions_by_target_port.values()) + "]"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))

###############################################################################
def _runtime_constant(context: NodeRuntimeContext) -> Any:
    parameters = context.normalized_parameters
    layer = context.runtime_services["constant_source"](
        shape=list(parameters.get("shape", [1])),
        value=parameters.get("value", 0),
        value_dtype=parameters.get("dtype", "float32"),
        name=context.node_id,
    )
    anchor = context.runtime_services.get("anchor")
    return layer(anchor) if anchor is not None else layer()

###############################################################################
def _runtime_parameter(context: NodeRuntimeContext) -> Any:
    parameters = context.normalized_parameters
    layer = context.runtime_services["parameter_source"](
        shape=list(parameters.get("shape", [1])),
        initializer=parameters.get("initializer", "zeros"),
        value_dtype=parameters.get("dtype", "float32"),
        trainable=bool(parameters.get("trainable", True)),
        name=context.node_id,
    )
    anchor = context.runtime_services.get("anchor")
    return layer(anchor) if anchor is not None else layer()

###############################################################################
def _runtime_index(context: NodeRuntimeContext) -> Any:
    expression = str(context.normalized_parameters.get("expression", ":"))
    return context.runtime_services["evaluate_safe_index_expression"](
        expression, context.source
    )

###############################################################################
def _runtime_average(context: NodeRuntimeContext) -> Any:
    ops = context.runtime_services["keras_ops"]
    return ops.sum(ops.stack(list(context.inputs), axis=0), axis=0) / len(context.inputs)

###############################################################################
def _runtime_split(context: NodeRuntimeContext) -> Any:
    parameters = context.normalized_parameters
    return context.runtime_services["keras_ops"].split(
        context.source,
        int(parameters.get("sections", 2)),
        axis=int(parameters.get("axis", -1)),
    )

###############################################################################
def _runtime_detection(context: NodeRuntimeContext) -> Any:
    return list(context.inputs)

###############################################################################
def _reshape(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    target_shape = tuple(int(value) for value in parameters.get("target_shape", [-1]))
    expression = (
        f"keras.layers.Reshape({target_shape!r}, "
        f"dtype={parameters.get('dtype')!r}, "
        f"trainable={bool(parameters.get('trainable', True))!r}, "
        f"autocast={bool(parameters.get('autocast', True))!r}, "
        f"name={context.canonical_ir_node.variable_name!r})({source})"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _permute(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    axes = tuple(int(value) for value in parameters.get("axes", [2, 1]))
    expression = (
        f"keras.layers.Permute({axes!r}, "
        f"dtype={parameters.get('dtype')!r}, "
        f"trainable={bool(parameters.get('trainable', True))!r}, "
        f"autocast={bool(parameters.get('autocast', True))!r}, "
        f"name={context.canonical_ir_node.variable_name!r})({source})"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _repeat(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    repeats = int(parameters.get("repeats", 2))
    axis = int(parameters.get("axis", 1))
    if axis == 1:
        expression = (
            f"keras.layers.RepeatVector({repeats}, "
            f"dtype={parameters.get('dtype')!r}, "
            f"trainable={bool(parameters.get('trainable', True))!r}, "
            f"autocast={bool(parameters.get('autocast', True))!r}, "
            f"name={context.canonical_ir_node.variable_name!r})({source})"
        )
    else:
        expression = f"keras.ops.repeat({source}, {repeats}, axis={axis})"
        if parameters.get("dtype") is not None:
            expression = f"keras.ops.cast({expression}, {parameters['dtype']!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("keras")
    authoritative_adapters = {
        "constant",
        "parameter",
        "index",
        "average",
        "split",
        "detection",
        "keras_reshape",
        "keras_permute",
        "keras_repeat",
    }
    runtime_emitters = {
        "constant": _runtime_constant,
        "parameter": _runtime_parameter,
        "index": _runtime_index,
        "average": _runtime_average,
        "split": _runtime_split,
        "detection": _runtime_detection,
    }
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
        registry.register(
            adapter,
            emitter,
            runtime=runtime_emitters.get(adapter),
            authoritative=adapter in authoritative_adapters,
        )
    registry.declare_parameter_ids("constant", {"shape", "value", "dtype"})
    registry.declare_parameter_ids(
        "parameter", {"shape", "initializer", "dtype", "trainable"}
    )
    registry.declare_parameter_ids("average", set())
    registry.declare_parameter_ids("split", {"sections", "axis"})
    registry.declare_parameter_ids("detection", set())
    registry.declare_parameter_ids("index", {"expression"})
    registry.declare_parameter_ids(
        "keras_reshape", {"target_shape", "dtype", "trainable", "autocast"}
    )
    registry.declare_parameter_ids(
        "keras_permute", {"axes", "dtype", "trainable", "autocast"}
    )
    registry.declare_parameter_ids(
        "keras_repeat", {"repeats", "axis", "dtype", "trainable", "autocast"}
    )
    return registry
