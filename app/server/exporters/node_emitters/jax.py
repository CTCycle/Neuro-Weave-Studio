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
    expression = safe_index_expression(context.normalized_parameters.get("expression", ":"))
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
def _activation(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    activation = str(context.normalized_parameters.get("activation", "relu"))
    function = {
        "linear": "lambda value: value",
        "relu": "nn.relu",
        "sigmoid": "nn.sigmoid",
        "tanh": "nn.tanh",
        "gelu": "nn.gelu",
        "softmax": "(lambda value: nn.softmax(value, axis=-1))",
    }.get(activation)
    if function is None:
        raise ValueError(f"No native JAX activation adapter registered for {activation}")
    expression = f"({function})({source})"
    dtype = context.normalized_parameters.get("dtype")
    if dtype is not None:
        expression = f"jnp.asarray({expression}, dtype=jnp.dtype({dtype!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _dropout(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = (
        f"nn.Dropout(rate={float(parameters.get('rate', 0.5))!r}, "
        f"deterministic=True, name={context.canonical_ir_node.variable_name!r})({source})"
    )
    dtype = parameters.get("dtype")
    if dtype is not None:
        expression = f"jnp.asarray({expression}, dtype=jnp.dtype({dtype!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _flatten(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    expression = f"jnp.reshape({source}, ({source}.shape[0], -1))"
    dtype = context.normalized_parameters.get("dtype")
    if dtype is not None:
        expression = f"jnp.asarray({expression}, dtype=jnp.dtype({dtype!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _conv1d(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    name = context.canonical_ir_node.variable_name
    data_format = str(parameters.get("data_format", "channels_last"))
    input_name = f"{name}_input"
    setup = [
        f"{input_name} = {source}.transpose(0, 2, 1)"
        if data_format == "channels_first"
        else f"{input_name} = {source}",
    ]
    kernel_size = int(parameters.get("kernel_size", [3])[0])
    stride = int(parameters.get("strides", [1])[0])
    dilation = int(parameters.get("dilation_rate", [1])[0])
    padding = str(parameters.get("padding", "same")).upper()
    setup.append(
        f"{name} = nn.Conv(features={int(parameters.get('filters', 32))}, "
        f"kernel_size={(kernel_size,)!r}, strides={(stride,)!r}, "
        f"kernel_dilation={(dilation,)!r}, padding={padding!r}, "
        f"feature_group_count={int(parameters.get('groups', 1))}, "
        f"use_bias={bool(parameters.get('use_bias', True))!r}, name={name!r})({input_name})"
    )
    if data_format == "channels_first":
        setup.append(f"{name} = {name}.transpose(0, 2, 1)")
    activation = str(parameters.get("activation", "linear"))
    expression = name
    if activation != "linear":
        function = {
            "relu": "nn.relu",
            "gelu": "nn.gelu",
            "sigmoid": "nn.sigmoid",
            "tanh": "nn.tanh",
            "softmax": "(lambda value: nn.softmax(value, axis=-1))",
        }.get(activation)
        if function is None:
            raise ValueError(f"No native JAX activation adapter registered for {activation}")
        expression = f"({function})({name})"
    if parameters.get("dtype") is not None:
        expression = f"jnp.asarray({expression}, dtype=jnp.dtype({parameters['dtype']!r}))"
    return EmissionResult(
        setup_statements=tuple(setup),
        output_expressions={context.materialized_output_ports[0].id: expression},
    )


###############################################################################
def _reshape(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    target_shape = tuple(int(value) for value in context.normalized_parameters.get("target_shape", [-1]))
    expression = f"jnp.reshape({source}, ({source}.shape[0],) + {target_shape!r})"
    if context.normalized_parameters.get("dtype") is not None:
        expression = f"jnp.asarray({expression}, dtype=jnp.dtype({context.normalized_parameters['dtype']!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def _permute(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    axes = tuple(int(value) for value in context.normalized_parameters.get("axes", [2, 1]))
    expression = f"jnp.transpose({source}, (0,) + {axes!r})"
    if context.normalized_parameters.get("dtype") is not None:
        expression = f"jnp.asarray({expression}, dtype=jnp.dtype({context.normalized_parameters['dtype']!r}))"
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
    if parameters.get("dtype") is not None:
        expression = f"jnp.asarray({expression}, dtype=jnp.dtype({parameters['dtype']!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


###############################################################################
def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("jax")
    authoritative_adapters = {
        "constant",
        "parameter",
        "index",
        "average",
        "split",
        "detection",
        "activation",
        "dropout",
        "flatten",
        "conv1d",
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
        "activation": _activation,
        "dropout": _dropout,
        "flatten": _flatten,
        "conv1d": _conv1d,
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
    registry.declare_parameter_ids("activation", {"activation", "dtype"})
    registry.declare_parameter_ids("dropout", {"rate", "dtype"})
    registry.declare_parameter_ids("flatten", {"data_format", "dtype"})
    registry.declare_parameter_ids(
        "conv1d",
        {
            "dtype",
            "strides",
            "padding",
            "data_format",
            "dilation_rate",
            "filters",
            "kernel_size",
            "activation",
            "use_bias",
            "groups",
        },
    )
    registry.declare_parameter_ids("reshape", {"target_shape", "dtype"})
    registry.declare_parameter_ids("permute", {"axes", "dtype"})
    registry.declare_parameter_ids("repeat", {"repeats", "axis", "dtype"})
    return registry
