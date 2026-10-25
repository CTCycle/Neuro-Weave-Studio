# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

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
    expression = f"torch.as_tensor({parameters.get('value', 0)!r}, dtype=getattr(torch, {parameters.get('dtype', 'float32')!r})).expand({tuple(parameters.get('shape', [1]))!r})"
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
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"torch.stack([{', '.join(inputs)}], dim=0).sum(dim=0) / {len(inputs)}"})

###############################################################################
def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"torch.tensor_split({source}, {int(parameters.get('sections', 2))}, dim={int(parameters.get('axis', -1))})"
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
        "relu": "torch.relu",
        "sigmoid": "torch.sigmoid",
        "tanh": "torch.tanh",
        "gelu": "torch.nn.functional.gelu",
        "softmax": "lambda value: torch.softmax(value, dim=-1)",
    }.get(activation)
    if function is None:
        raise ValueError(f"No PyTorch activation adapter registered for {activation}")
    expression = f"({function})({source})"
    dtype = context.normalized_parameters.get("dtype")
    if dtype is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {dtype!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _dropout(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    expression = f"self.{context.canonical_ir_node.variable_name}({source})"
    dtype = context.normalized_parameters.get("dtype")
    if dtype is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {dtype!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _flatten(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    expression = f"self.{context.canonical_ir_node.variable_name}({source})"
    dtype = context.normalized_parameters.get("dtype")
    if dtype is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {dtype!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _conv1d(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    name = context.canonical_ir_node.variable_name
    kernel_size = int(parameters.get("kernel_size", [3])[0])
    stride = int(parameters.get("strides", [1])[0])
    dilation = int(parameters.get("dilation_rate", [1])[0])
    padding = str(parameters.get("padding", "same"))
    data_format = str(parameters.get("data_format", "channels_last"))
    conv_input = f"{name}_input"
    setup = [f"{conv_input} = {source}"]
    if data_format == "channels_last":
        setup.append(f"{conv_input} = {conv_input}.permute(0, 2, 1).contiguous()")
    if padding == "same":
        setup.append(
            f"{conv_input} = _same_pad_1d({conv_input}, {kernel_size!r}, {stride!r}, {dilation!r})"
        )
    setup.append(f"{name} = self.{name}({conv_input})")
    if data_format == "channels_last":
        setup.append(f"{name} = {name}.permute(0, 2, 1).contiguous()")
    activation = str(parameters.get("activation", "linear"))
    expression = name
    if activation != "linear":
        function = {
            "relu": "torch.relu",
            "gelu": "torch.nn.functional.gelu",
            "sigmoid": "torch.sigmoid",
            "tanh": "torch.tanh",
            "softmax": "lambda value: torch.softmax(value, dim=-1)",
        }.get(activation)
        if function is None:
            raise ValueError(f"No PyTorch activation adapter registered for {activation}")
        expression = f"({function})({name})"
    if parameters.get("dtype") is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {parameters['dtype']!r}))"
    return EmissionResult(
        setup_statements=tuple(setup),
        output_expressions={context.materialized_output_ports[0].id: expression},
    )

###############################################################################
def _conv2d(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    name = context.canonical_ir_node.variable_name
    kernel_size = tuple(int(value) for value in parameters.get("kernel_size", [3, 3]))
    strides = tuple(int(value) for value in parameters.get("strides", [1, 1]))
    dilation = tuple(int(value) for value in parameters.get("dilation_rate", [1, 1]))
    padding = str(parameters.get("padding", "same"))
    data_format = str(parameters.get("data_format", "channels_last"))
    conv_input = f"{name}_input"
    setup = [f"{conv_input} = {source}"]
    if data_format == "channels_last":
        setup.append(f"{conv_input} = {conv_input}.permute(0, 3, 1, 2).contiguous()")
    if padding == "same":
        setup.append(
            f"{conv_input} = _same_pad_2d({conv_input}, {kernel_size!r}, {strides!r}, {dilation!r})"
        )
    setup.append(f"{name} = self.{name}({conv_input})")
    if data_format == "channels_last":
        setup.append(f"{name} = {name}.permute(0, 2, 3, 1).contiguous()")
    activation = str(parameters.get("activation", "linear"))
    expression = name
    if activation != "linear":
        function = {
            "relu": "torch.relu",
            "gelu": "torch.nn.functional.gelu",
            "sigmoid": "torch.sigmoid",
            "tanh": "torch.tanh",
            "softmax": "lambda value: torch.softmax(value, dim=-1)",
        }.get(activation)
        if function is None:
            raise ValueError(f"No PyTorch activation adapter registered for {activation}")
        expression = f"({function})({name})"
    if parameters.get("dtype") is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {parameters['dtype']!r}))"
    return EmissionResult(
        setup_statements=tuple(setup),
        output_expressions={context.materialized_output_ports[0].id: expression},
    )

###############################################################################
def _conv3d(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    name = context.canonical_ir_node.variable_name
    kernel_size = tuple(int(value) for value in parameters.get("kernel_size", [3, 3, 3]))
    strides = tuple(int(value) for value in parameters.get("strides", [1, 1, 1]))
    dilation = tuple(int(value) for value in parameters.get("dilation_rate", [1, 1, 1]))
    padding = str(parameters.get("padding", "same"))
    data_format = str(parameters.get("data_format", "channels_last"))
    conv_input = f"{name}_input"
    setup = [f"{conv_input} = {source}"]
    if data_format == "channels_last":
        setup.append(
            f"{conv_input} = {conv_input}.permute((0, 4, 1, 2, 3)).contiguous()"
        )
    if padding == "same":
        setup.append(
            f"{conv_input} = _same_pad_nd({conv_input}, {kernel_size!r}, "
            f"{strides!r}, 0.0, {dilation!r})"
        )
    setup.append(f"{name} = self.{name}({conv_input})")
    if data_format == "channels_last":
        setup.append(
            f"{name} = {name}.permute((0, 2, 3, 4, 1)).contiguous()"
        )
    activation = str(parameters.get("activation", "linear"))
    expression = name
    if activation != "linear":
        function = {
            "relu": "torch.relu",
            "gelu": "torch.nn.functional.gelu",
            "sigmoid": "torch.sigmoid",
            "tanh": "torch.tanh",
            "softmax": "lambda value: torch.softmax(value, dim=-1)",
        }.get(activation)
        if function is None:
            raise ValueError(f"No PyTorch activation adapter registered for {activation}")
        expression = f"({function})({name})"
    if parameters.get("dtype") is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {parameters['dtype']!r}))"
    return EmissionResult(
        setup_statements=tuple(setup),
        output_expressions={context.materialized_output_ports[0].id: expression},
    )

###############################################################################
def _dense(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    name = context.canonical_ir_node.variable_name
    expression = f"self.{name}({source})"
    activation = str(parameters.get("activation", "linear"))
    if activation != "linear":
        function = {
            "relu": "torch.relu",
            "gelu": "torch.nn.functional.gelu",
            "sigmoid": "torch.sigmoid",
            "tanh": "torch.tanh",
            "softmax": "lambda value: torch.softmax(value, dim=-1)",
        }.get(activation)
        if function is None:
            raise ValueError(f"No PyTorch activation adapter registered for {activation}")
        expression = f"({function})({expression})"
    if parameters.get("dtype") is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {parameters['dtype']!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _reshape(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    target_shape = tuple(int(value) for value in context.normalized_parameters.get("target_shape", [-1]))
    expression = f"torch.reshape({source}, ({source}.shape[0],) + {target_shape!r})"
    if context.normalized_parameters.get("dtype") is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {context.normalized_parameters['dtype']!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _permute(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    axes = tuple(int(value) for value in context.normalized_parameters.get("axes", [2, 1]))
    expression = f"{source}.permute((0,) + {axes!r})"
    if context.normalized_parameters.get("dtype") is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {context.normalized_parameters['dtype']!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def _repeat(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    repeats = int(parameters.get("repeats", 2))
    axis = int(parameters.get("axis", 1))
    expression = (
        f"{source}.unsqueeze(1).repeat(1, {repeats}, 1)"
        if axis == 1
        else f"torch.repeat_interleave({source}, {repeats}, dim={axis})"
    )
    if parameters.get("dtype") is not None:
        expression = f"({expression}).to(dtype=getattr(torch, {parameters['dtype']!r}))"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})

###############################################################################
def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("pytorch")
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
        "conv2d",
        "conv3d",
        "dense",
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
        "conv2d": _conv2d,
        "conv3d": _conv3d,
        "dense": _dense,
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
    registry.declare_parameter_ids(
        "conv2d",
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
    registry.declare_parameter_ids(
        "conv3d",
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
    registry.declare_parameter_ids("dense", {"units", "activation", "use_bias", "dtype"})
    registry.declare_parameter_ids("reshape", {"target_shape", "dtype"})
    registry.declare_parameter_ids("permute", {"axes", "dtype"})
    registry.declare_parameter_ids("repeat", {"repeats", "axis", "dtype"})
    return registry
