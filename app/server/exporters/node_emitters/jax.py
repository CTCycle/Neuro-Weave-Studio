from __future__ import annotations

from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    emit_multi_output_assignment,
)


###############################################################################
def _simple(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()), "")
    return EmissionResult(output_expressions=emit_multi_output_assignment(source, context.materialized_output_ports))


###############################################################################
def _constant(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = f"jnp.broadcast_to(jnp.asarray({parameters.get('value', 0)!r}, dtype=jnp.dtype({parameters.get('dtype', 'float32')!r})), {tuple(parameters.get('shape', [1]))!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _parameter(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    initializer = str(parameters.get("initializer", "zeros"))
    initializer_name = {
        "zeros": "zeros",
        "ones": "ones",
        "uniform": "uniform",
        "normal": "normal",
    }.get(initializer, "zeros")
    initializer_expression = f"nn.initializers.{initializer_name}"
    shape = tuple(parameters.get("shape", [1]))
    dtype = parameters.get("dtype", "float32")
    name = context.canonical_ir_node.variable_name
    if bool(parameters.get("trainable", True)):
        expression = (
            f"self.param({name!r}, {initializer_expression}, {shape!r}, "
            f"jnp.dtype({dtype!r}))"
        )
    else:
        expression = (
            f"self.variable('constants', {name!r}, lambda: "
            f"jnp.zeros({shape!r}, dtype=jnp.dtype({dtype!r}))).value"
        )
    return EmissionResult(
        output_expressions={context.materialized_output_ports[0].id: expression}
    )


###############################################################################
def _index(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    value = context.normalized_parameters.get("expression", ":")
    expression = value if isinstance(value, str) and value.startswith("x") else f"x[{value}]"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression.replace("x", source, 1)})


###############################################################################
def _average(context: NodeEmitterContext) -> EmissionResult:
    inputs = list(context.input_expressions_by_target_port.values())
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"jnp.stack([{', '.join(inputs)}], axis=0).sum(axis=0) / {len(inputs)}"})


###############################################################################
def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"jnp.split({source}, {int(parameters.get('sections', 2))}, axis={int(parameters.get('axis', -1))})"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


###############################################################################
def _detection(context: NodeEmitterContext) -> EmissionResult:
    expression = "(" + ", ".join(context.input_expressions_by_target_port.values()) + ")"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


###############################################################################
def _reshape(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    target_shape = tuple(int(value) for value in context.normalized_parameters.get("target_shape", [-1]))
    expression = f"jnp.reshape({source}, ({source}.shape[0],) + {target_shape!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _permute(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    axes = tuple(int(value) for value in context.normalized_parameters.get("axes", [2, 1]))
    expression = f"jnp.transpose({source}, (0,) + {axes!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _repeat(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    repeats = int(parameters.get("repeats", 2))
    axis = int(parameters.get("axis", 1))
    expression = (
        f"jnp.repeat({source}[:, None, :], {repeats}, axis=1)"
        if axis == 1
        else f"jnp.repeat({source}, {repeats}, axis={axis})"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("jax")
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
        registry.register(adapter, emitter)
    registry.declare_parameter_ids("constant", {"shape", "value", "dtype"})
    registry.declare_parameter_ids(
        "parameter", {"shape", "initializer", "dtype", "trainable"}
    )
    registry.declare_parameter_ids("average", set())
    registry.declare_parameter_ids("split", {"sections", "axis"})
    registry.declare_parameter_ids("detection", set())
    return registry
