# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from typing import Any, cast

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import CompilationCatalog
from domain.graph import ModelGraph
from domain.ports import materialized_outputs
from domain.tensor import TensorSpec
from exporters.base import SourceExporter
from validation.safe_expression import normalize_index_expression, validate_safe_expression


###############################################################################
class NativeTensorFlowSourceExporter(SourceExporter):
    """Generate a native TensorFlow ``tf.Module`` for the verified subset."""

    # -------------------------------------------------------------------------
    @staticmethod
    def _summary_map(ir: CanonicalModelIR) -> dict[str, TensorSpec]:
        return {
            summary.node_id: next(iter(summary.outputs.values()))
            for summary in ir.shape_summaries
            if summary.outputs
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _static_dimension(spec: TensorSpec | None, index: int) -> int | None:
        if spec is None:
            return None
        shape = cast(list[Any], spec.shape)
        if not -len(shape) <= index < len(shape):
            return None
        value = shape[index]
        return value if isinstance(value, int) else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _spatial_tuple(parameters: dict[str, Any], key: str, default: list[int]) -> tuple[int, int]:
        value = parameters.get(key, default)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"TensorFlow {key} expects two values")
        values = cast(list[Any], value)
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in values):
            raise ValueError(f"TensorFlow {key} expects positive integer values")
        return int(values[0]), int(values[1])

    # -------------------------------------------------------------------------
    @staticmethod
    def _spatial_values(
        parameters: dict[str, Any],
        key: str,
        dimensions: int,
        default: list[int],
    ) -> tuple[int, ...]:
        value = parameters.get(key, default)
        if value is None:
            value = default
        if not isinstance(value, list) or len(value) != dimensions:
            raise ValueError(f"TensorFlow {key} expects {dimensions} values")
        values = cast(list[Any], value)
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in values):
            raise ValueError(f"TensorFlow {key} expects positive integer values")
        return tuple(int(item) for item in values)

    # -------------------------------------------------------------------------
    @staticmethod
    def _spatial_nonnegative(
        parameters: dict[str, Any],
        key: str,
        dimensions: int,
        default: list[int],
    ) -> tuple[int, ...]:
        value = parameters.get(key, default)
        if value is None:
            value = default
        if not isinstance(value, list) or len(value) != dimensions:
            raise ValueError(f"TensorFlow {key} expects {dimensions} values")
        values = cast(list[Any], value)
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
            raise ValueError(f"TensorFlow {key} expects non-negative integer values")
        return tuple(int(item) for item in values)

    # -------------------------------------------------------------------------
    @staticmethod
    def _spatial_pairs(parameters: dict[str, Any], key: str, dimensions: int) -> tuple[tuple[int, int], ...]:
        value = parameters.get(key)
        if not isinstance(value, list) or len(value) != dimensions:
            raise ValueError(f"TensorFlow {key} expects {dimensions} pairs")
        pairs: list[tuple[int, int]] = []
        for item in cast(list[Any], value):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(not isinstance(entry, int) or isinstance(entry, bool) or entry < 0 for entry in item)
            ):
                raise ValueError(f"TensorFlow {key} expects non-negative pairs")
            pairs.append((int(item[0]), int(item[1])))
        return tuple(pairs)

    # -------------------------------------------------------------------------
    @staticmethod
    def _axis(value: object, rank: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("TensorFlow normalization expects an integer axis")
        axis = value if value >= 0 else rank + value
        if axis < 0 or axis >= rank:
            raise ValueError(f"Normalization axis {value} is outside rank {rank}")
        return axis

    # -------------------------------------------------------------------------
    @staticmethod
    def _index_expression(value: object) -> str:
        expression = normalize_index_expression(value if isinstance(value, str) else ":")
        validate_safe_expression(expression)
        return expression

    # -------------------------------------------------------------------------
    @staticmethod
    def _condition_matches(condition: dict[str, object], parameters: dict[str, object]) -> bool:
        for parameter_id, expected in condition.items():
            actual = parameters.get(parameter_id)
            if isinstance(expected, dict):
                if "equals" in expected and actual != expected["equals"]:
                    return False
                if "not_equals" in expected and actual == expected["not_equals"]:
                    return False
                if "in" in expected and actual not in expected["in"]:
                    return False
                if "not_in" in expected and actual in expected["not_in"]:
                    return False
            elif (
                isinstance(expected, list) and actual not in expected
            ) or (not isinstance(expected, (dict, list)) and actual != expected):
                return False
        return True

    # -------------------------------------------------------------------------
    @classmethod
    def _normalization_axes(cls, value: object, rank: int) -> list[int]:
        if not isinstance(value, list) or not value:
            raise ValueError("TensorFlow LayerNorm expects a non-empty axis list")
        values = cast(list[Any], value)
        axes = [cls._axis(item, rank) for item in values]
        if axes != list(range(rank - len(axes), rank)):
            raise ValueError(
                "TensorFlow LayerNorm requires normalized axes at the end of the tensor"
            )
        return axes

    # -------------------------------------------------------------------------
    def generate(
        self,
        graph: ModelGraph,
        ir: CanonicalModelIR,
        catalog: NodeCatalog | None = None,
        *,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> str:
        del compilation_catalog
        catalog = catalog or get_node_catalog()
        nodes = {node.id: node for node in graph.nodes}
        summaries = self._summary_map(ir)
        lines = [
            "from __future__ import annotations",
            "",
            "import tensorflow as tf",
            "",
            "",
            "def _activation(value, name):",
            "    if name == 'linear': return value",
            "    if name == 'relu': return tf.nn.relu(value)",
            "    if name == 'sigmoid': return tf.nn.sigmoid(value)",
            "    if name == 'tanh': return tf.nn.tanh(value)",
            "    if name == 'gelu': return tf.nn.gelu(value)",
            "    if name == 'softmax': return tf.nn.softmax(value, axis=-1)",
            "    raise ValueError(f'Unsupported TensorFlow activation: {name}')",
            "",
            "",
            "def _same_pad(value, kernel_size, strides, dilation, fill_value=0.0):",
            "    shape = tf.shape(value)",
            "    output_height = (shape[1] + strides[0] - 1) // strides[0]",
            "    output_width = (shape[2] + strides[1] - 1) // strides[1]",
            "    effective_height = dilation[0] * (kernel_size[0] - 1) + 1",
            "    effective_width = dilation[1] * (kernel_size[1] - 1) + 1",
            "    total_height = tf.maximum((output_height - 1) * strides[0] + effective_height - shape[1], 0)",
            "    total_width = tf.maximum((output_width - 1) * strides[1] + effective_width - shape[2], 0)",
            "    top = total_height // 2",
            "    left = total_width // 2",
            "    paddings = [[0, 0], [top, total_height - top], [left, total_width - left], [0, 0]]",
            "    return tf.pad(value, paddings, constant_values=tf.cast(fill_value, value.dtype))",
            "",
            "",
            "def _conv2d(value, kernel, bias, strides, dilation, padding, data_format):",
            "    channels_first = data_format == 'channels_first'",
            "    current = tf.transpose(value, [0, 2, 3, 1]) if channels_first else value",
            "    if padding == 'same': current = _same_pad(current, kernel.shape[:2], strides, dilation)",
            "    current = tf.nn.conv2d(current, kernel, strides=[1, strides[0], strides[1], 1], dilations=[1, dilation[0], dilation[1], 1], padding='VALID' if padding == 'same' else padding.upper())",
            "    if bias is not None: current = tf.nn.bias_add(current, bias)",
            "    return tf.transpose(current, [0, 3, 1, 2]) if channels_first else current",
            "",
            "",
            "def _conv3d(value, kernel, bias, strides, dilation, padding, data_format, groups):",
            "    channels_first = data_format == 'channels_first'",
            "    current = tf.transpose(value, [0, 2, 3, 4, 1]) if channels_first else value",
            "    stride_value = [1, strides[0], strides[1], strides[2], 1]",
            "    dilation_value = [1, dilation[0], dilation[1], dilation[2], 1]",
            "    if groups == 1:",
            "        current = tf.nn.conv3d(current, kernel, strides=stride_value, dilations=dilation_value, padding=padding.upper(), data_format='NDHWC')",
            "    else:",
            "        current = tf.concat([tf.nn.conv3d(part, kernel_part, strides=stride_value, dilations=dilation_value, padding=padding.upper(), data_format='NDHWC') for part, kernel_part in zip(tf.split(current, groups, axis=4), tf.split(kernel, groups, axis=4))], axis=4)",
            "    if bias is not None: current = tf.nn.bias_add(current, bias, data_format='NDHWC')",
            "    return tf.transpose(current, [0, 4, 1, 2, 3]) if channels_first else current",
            "",
            "",
            "def _depthwise(value, kernel, bias, strides, dilation, padding, data_format, dimensions):",
            "    channels_first = data_format == 'channels_first'",
            "    if dimensions == 1:",
            "        current = tf.transpose(value, [0, 2, 1]) if channels_first else value",
            "        current = tf.expand_dims(current, axis=1)",
            "        current = tf.raw_ops.DepthwiseConv2dNative(input=current, filter=kernel[None, ...], strides=[1, 1, strides[0], 1], dilations=[1, 1, dilation[0], 1], padding=padding.upper())",
            "        current = tf.squeeze(current, axis=1)",
            "        if bias is not None: current = tf.nn.bias_add(current, bias, data_format='NWC')",
            "        return tf.transpose(current, [0, 2, 1]) if channels_first else current",
            "    current = tf.transpose(value, [0, 2, 3, 1]) if channels_first else value",
            "    current = tf.raw_ops.DepthwiseConv2dNative(input=current, filter=kernel, strides=[1, strides[0], strides[1], 1], dilations=[1, dilation[0], dilation[1], 1], padding=padding.upper())",
            "    if bias is not None: current = tf.nn.bias_add(current, bias, data_format='NHWC')",
            "    return tf.transpose(current, [0, 3, 1, 2]) if channels_first else current",
            "",
            "",
            "def _separable(value, depthwise_kernel, pointwise_kernel, bias, strides, dilation, padding, data_format, dimensions):",
            "    current = _depthwise(value, depthwise_kernel, None, strides, dilation, padding, data_format, dimensions)",
            "    if dimensions == 1:",
            "        return _conv1d(current, pointwise_kernel, bias, 1, 1, 'valid', data_format, 1)",
            "    return _conv2d(current, pointwise_kernel, bias, (1, 1), (1, 1), 'valid', data_format)",
            "",
            "",
            "def _transposed(value, kernel, bias, filters, strides, dilation, padding, output_padding, data_format, dimensions):",
            "    channels_first = data_format == 'channels_first'",
            "    if channels_first:",
            "        value = tf.transpose(value, [0, *range(2, dimensions + 2), 1])",
            "    input_shape = tf.shape(value)",
            "    output_spatial = []",
            "    for index in range(dimensions):",
            "        effective_kernel = dilation[index] * (kernel.shape[index] - 1) + 1",
            "        output_size = input_shape[index + 1] * strides[index] if padding == 'same' else (input_shape[index + 1] - 1) * strides[index] + effective_kernel",
            "        if output_padding is not None: output_size = output_size + output_padding[index]",
            "        output_spatial.append(output_size)",
            "    output_shape = tf.stack([input_shape[0], *output_spatial, filters])",
            "    operation = getattr(tf.nn, f'conv{dimensions}d_transpose')",
            "    strides_value = strides[0] if dimensions == 1 else [1, *strides, 1]",
            "    dilation_value = dilation[0] if dimensions == 1 else [1, *dilation, 1]",
            "    current = operation(value, kernel, output_shape=output_shape, strides=strides_value, padding=padding.upper(), dilations=dilation_value, data_format={1: 'NWC', 2: 'NHWC', 3: 'NDHWC'}[dimensions])",
            "    if bias is not None: current = tf.nn.bias_add(current, bias, data_format={1: 'NWC', 2: 'NHWC', 3: 'NDHWC'}[dimensions])",
            "    if channels_first: current = tf.transpose(current, [0, dimensions + 1, *range(1, dimensions + 1)])",
            "    return current",
            "",
            "",
            "def _same_pad_1d(value, kernel_size, stride, dilation, fill_value=0.0):",
            "    length = tf.shape(value)[1]",
            "    output_length = (length + stride - 1) // stride",
            "    effective_kernel = dilation * (kernel_size - 1) + 1",
            "    total = tf.maximum((output_length - 1) * stride + effective_kernel - length, 0)",
            "    left = total // 2",
            "    paddings = [[0, 0], [left, total - left], [0, 0]]",
            "    return tf.pad(value, paddings, constant_values=tf.cast(fill_value, value.dtype))",
            "",
            "",
            "def _conv1d(value, kernel, bias, stride, dilation, padding, data_format, groups):",
            "    channels_first = data_format == 'channels_first'",
            "    current = tf.transpose(value, [0, 2, 1]) if channels_first else value",
            "    if padding == 'same': current = _same_pad_1d(current, kernel.shape[0], stride, dilation)",
            "    if groups == 1:",
            "        current = tf.nn.conv1d(current, kernel, stride=stride, dilations=dilation, padding='VALID' if padding == 'same' else padding.upper(), data_format='NWC')",
            "    else:",
            "        current = tf.concat([tf.nn.conv1d(part, kernel_part, stride=stride, dilations=dilation, padding='VALID' if padding == 'same' else padding.upper(), data_format='NWC') for part, kernel_part in zip(tf.split(current, groups, axis=2), tf.split(kernel, groups, axis=2))], axis=2)",
            "    if bias is not None: current = tf.nn.bias_add(current, bias, data_format='NWC')",
            "    return tf.transpose(current, [0, 2, 1]) if channels_first else current",
            "",
            "",
            "def _max_pool2d(value, pool_size, strides, padding, data_format):",
            "    channels_first = data_format == 'channels_first'",
            "    current = tf.transpose(value, [0, 2, 3, 1]) if channels_first else value",
            "    if padding == 'same': current = _same_pad(current, pool_size, strides, (1, 1), float('-inf'))",
            "    current = tf.nn.max_pool2d(current, ksize=pool_size, strides=strides, padding='VALID' if padding == 'same' else padding.upper(), data_format='NHWC')",
            "    return tf.transpose(current, [0, 3, 1, 2]) if channels_first else current",
            "",
            "",
            "def _pool_nd(value, pool_size, strides, padding, data_format, mode):",
            "    dimensions = len(pool_size)",
            "    channels_first = data_format == 'channels_first'",
            "    if channels_first:",
            "        value = tf.transpose(value, [0, *range(2, dimensions + 2), 1])",
            "    operation = getattr(tf.nn, f'{mode}_pool{dimensions}d')",
            "    format_name = {1: 'NWC', 2: 'NHWC', 3: 'NDHWC'}[dimensions]",
            "    current = operation(value, ksize=pool_size, strides=strides, padding=padding.upper(), data_format=format_name)",
            "    if channels_first:",
            "        current = tf.transpose(current, [0, dimensions + 1, *range(1, dimensions + 1)])",
            "    return current",
            "",
            "",
            "def _global_pool_nd(value, dimensions, data_format, mode, keepdims):",
            "    channels_first = data_format == 'channels_first'",
            "    if channels_first:",
            "        value = tf.transpose(value, [0, *range(2, dimensions + 2), 1])",
            "    spatial_axes = list(range(1, dimensions + 1))",
            "    operation = tf.reduce_mean if mode == 'average' else tf.reduce_max",
            "    current = operation(value, axis=spatial_axes, keepdims=keepdims)",
            "    if channels_first and keepdims:",
            "        current = tf.transpose(current, [0, dimensions + 1, *range(1, dimensions + 1)])",
            "    return current",
            "",
            "",
            "def _upsample_nd(value, size, data_format, interpolation):",
            "    dimensions = len(size)",
            "    channels_first = data_format == 'channels_first'",
            "    if channels_first:",
            "        value = tf.transpose(value, [0, *range(2, dimensions + 2), 1])",
            "    if interpolation == 'bilinear' and dimensions == 2:",
            "        target = [tf.shape(value)[1] * size[0], tf.shape(value)[2] * size[1]]",
            "        current = tf.image.resize(value, target, method='bilinear')",
            "    else:",
            "        current = value",
            "        for axis, multiplier in enumerate(size, start=1):",
            "            current = tf.repeat(current, multiplier, axis=axis)",
            "    if channels_first:",
            "        current = tf.transpose(current, [0, dimensions + 1, *range(1, dimensions + 1)])",
            "    return current",
            "",
            "",
            "def _pad_nd(value, pairs, data_format):",
            "    dimensions = len(pairs)",
            "    if data_format == 'channels_first':",
            "        paddings = [[0, 0], [0, 0], *pairs]",
            "    else:",
            "        paddings = [[0, 0], *pairs, [0, 0]]",
            "    return tf.pad(value, paddings)",
            "",
            "",
            "def _lstm(value, kernel, bias, units, return_sequences):",
            "    initial = (tf.zeros([tf.shape(value)[0], units], value.dtype), tf.zeros([tf.shape(value)[0], units], value.dtype))",
            "    def step(carry, current):",
            "        hidden, cell = carry",
            "        gates = tf.matmul(tf.concat([current, hidden], axis=-1), kernel)",
            "        if bias is not None: gates = gates + bias",
            "        input_gate, forget_gate, candidate, output_gate = tf.split(gates, 4, axis=-1)",
            "        next_cell = tf.sigmoid(forget_gate) * cell + tf.sigmoid(input_gate) * tf.tanh(candidate)",
            "        next_hidden = tf.sigmoid(output_gate) * tf.tanh(next_cell)",
            "        return next_hidden, next_cell",
            "    hidden_sequence, _ = tf.scan(step, tf.transpose(value, [1, 0, 2]), initializer=initial)",
            "    sequence = tf.transpose(hidden_sequence, [1, 0, 2])",
            "    return sequence if return_sequences else sequence[:, -1, :]",
            "",
            "",
            "def _gru(value, kernel, bias, units, return_sequences):",
            "    initial = tf.zeros([tf.shape(value)[0], units], value.dtype)",
            "    def step(hidden, current):",
            "        gates = tf.matmul(tf.concat([current, hidden], axis=-1), kernel)",
            "        if bias is not None: gates = gates + bias",
            "        update_gate, reset_gate, candidate = tf.split(gates, 3, axis=-1)",
            "        update_gate = tf.sigmoid(update_gate)",
            "        reset_gate = tf.sigmoid(reset_gate)",
            "        candidate = tf.tanh(candidate + reset_gate * hidden)",
            "        return (1.0 - update_gate) * candidate + update_gate * hidden",
            "    hidden_sequence = tf.scan(step, tf.transpose(value, [1, 0, 2]), initializer=initial)",
            "    sequence = tf.transpose(hidden_sequence, [1, 0, 2])",
            "    return sequence if return_sequences else sequence[:, -1, :]",
            "",
            "",
            "def _simple_rnn(value, kernel, bias, units, return_sequences):",
            "    initial = tf.zeros([tf.shape(value)[0], units], value.dtype)",
            "    def step(hidden, current):",
            "        state = tf.matmul(tf.concat([current, hidden], axis=-1), kernel)",
            "        if bias is not None: state = state + bias",
            "        return tf.tanh(state)",
            "    hidden_sequence = tf.scan(step, tf.transpose(value, [1, 0, 2]), initializer=initial)",
            "    sequence = tf.transpose(hidden_sequence, [1, 0, 2])",
            "    return sequence if return_sequences else sequence[:, -1, :]",
            "",
            "",
            "def _recurrent(value, cell, kernel, bias, units, return_sequences):",
            "    if cell == 'simple_rnn': return _simple_rnn(value, kernel, bias, units, return_sequences)",
            "    if cell == 'lstm': return _lstm(value, kernel, bias, units, return_sequences)",
            "    if cell == 'gru': return _gru(value, kernel, bias, units, return_sequences)",
            "    raise ValueError(f'Unsupported recurrent cell: {cell}')",
            "",
            "",
            "def _bidirectional(value, cell, forward_kernel, forward_bias, backward_kernel, backward_bias, units, merge_mode, return_sequences):",
            "    forward = _recurrent(value, cell, forward_kernel, forward_bias, units, True)",
            "    backward = _recurrent(tf.reverse(value, axis=[1]), cell, backward_kernel, backward_bias, units, True)",
            "    backward = tf.reverse(backward, axis=[1])",
            "    if merge_mode == 'sum': merged = forward + backward",
            "    elif merge_mode == 'mul': merged = forward * backward",
            "    elif merge_mode == 'ave': merged = (forward + backward) / 2.0",
            "    elif merge_mode == 'concat': merged = tf.concat([forward, backward], axis=-1)",
            "    else: raise ValueError(f'Unsupported bidirectional merge mode: {merge_mode}')",
            "    return merged if return_sequences else merged[:, -1, :]",
            "",
            "",
            "def _sinusoidal_encoding(value):",
            "    positions = tf.cast(tf.range(tf.shape(value)[1])[:, None], value.dtype)",
            "    indices = tf.cast(tf.range(tf.shape(value)[2])[None, :], value.dtype)",
            "    dimension = tf.cast(tf.shape(value)[2], value.dtype)",
            "    rates = tf.pow(tf.cast(10000.0, value.dtype), (2.0 * tf.floor(indices / 2.0)) / dimension)",
            "    angles = positions / rates",
            "    even = tf.equal(tf.math.floormod(tf.cast(indices, tf.int32), 2), 0)",
            "    encoding = tf.where(even, tf.sin(angles), tf.cos(angles))",
            "    return value + encoding[None, :, :]",
            "",
            "",
            "def _initialize(shape, initializer, dtype):",
            "    identifier = initializer.get('id', 'zeros') if isinstance(initializer, dict) else initializer",
            "    if identifier in (None, 'zeros'): return tf.zeros(shape, dtype=dtype)",
            "    if identifier == 'ones': return tf.ones(shape, dtype=dtype)",
            "    if identifier in ('random_normal', 'normal'): return tf.random.normal(shape, dtype=dtype)",
            "    if identifier in ('random_uniform', 'uniform'): return tf.random.uniform(shape, dtype=dtype)",
            "    return tf.zeros(shape, dtype=dtype)",
            "",
            "",
            "class GeneratedModel(tf.Module):",
            "    def __init__(self) -> None:",
            "        super().__init__(name='neuro_weave_model')",
        ]

        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            params = node.parameters
            definition = item.definition_id
            name = item.variable_name
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None
            if definition == "input.constant":
                shape = params.get("shape", [1])
                dtype = params.get("dtype", "float32")
                value = params.get("value", 0)
                lines.append(
                    f"        self.{name} = tf.broadcast_to(tf.cast(tf.convert_to_tensor({value!r}), {dtype!r}), {shape!r})"
                )
            elif definition == "input.parameter":
                shape = params.get("shape", [1])
                dtype = params.get("dtype", "float32")
                initializer = params.get("initializer", "zeros")
                lines.append(
                    f"        self.{name} = tf.Variable(_initialize({shape!r}, {initializer!r}, {dtype!r}), trainable={bool(params.get('trainable', True))!r}, name={name!r})"
                )
            elif definition == "layer.prelu":
                if source_spec is None:
                    raise ValueError(f"TensorFlow PReLU node {node.id} requires a known input shape")
                alpha_shape = list(source_spec.shape[1:])
                if not all(isinstance(value, int) for value in alpha_shape):
                    raise ValueError(f"TensorFlow PReLU node {node.id} requires known input dimensions")
                shared_axes = params.get("shared_axes")
                if shared_axes is not None:
                    if not isinstance(shared_axes, list):
                        raise ValueError("TensorFlow PReLU shared_axes must be a list or null")
                    for shared_axis in shared_axes:
                        if not isinstance(shared_axis, int) or not 1 <= shared_axis < source_spec.rank:
                            raise ValueError("TensorFlow PReLU shared_axes must contain non-batch axes")
                        alpha_shape[shared_axis - 1] = 1
                lines.append(
                    f"        self.{name}_alpha = tf.Variable(_initialize({alpha_shape!r}, {params.get('alpha_initializer', 'zeros')!r}, 'float32'), name={name + '_alpha'!r})"
                )
            elif definition == "layer.rms_norm":
                if source_spec is None:
                    raise ValueError(f"TensorFlow RMSNormalization node {node.id} requires a known input shape")
                axis = self._axis(params.get("axis", -1), source_spec.rank)
                dimension = self._static_dimension(source_spec, axis)
                if dimension is None:
                    raise ValueError(f"TensorFlow RMSNormalization node {node.id} requires a known normalized dimension")
                broadcast = [1] * source_spec.rank
                broadcast[axis] = dimension
                lines.append(f"        self.{name}_axis = {axis}")
                lines.append(f"        self.{name}_broadcast = {broadcast!r}")
                lines.append(
                    f"        self.{name}_epsilon = {float(params.get('epsilon', 1e-6))!r}"
                )
                lines.append(
                    f"        self.{name}_scale = tf.Variable(tf.ones([{dimension}], tf.float32), name={name + '_scale'!r})"
                )
            elif definition == "layer.unit_norm":
                if source_spec is None:
                    raise ValueError(f"TensorFlow UnitNormalization node {node.id} requires a known input rank")
                lines.append(f"        self.{name}_axis = {self._axis(params.get('axis', -1), source_spec.rank)}")
            elif definition == "layer.group_norm":
                if source_spec is None:
                    raise ValueError(f"TensorFlow GroupNormalization node {node.id} requires a known input shape")
                rank = source_spec.rank
                axis = self._axis(params.get("axis", -1), rank)
                channels = self._static_dimension(source_spec, axis)
                groups = int(params.get("groups", 32))
                if channels is None:
                    raise ValueError(f"TensorFlow GroupNormalization node {node.id} requires a known channel dimension")
                if channels % groups != 0:
                    raise ValueError("TensorFlow GroupNormalization channels must be divisible by groups")
                permutation = [index for index in range(rank) if index != axis] + [axis]
                inverse = [permutation.index(index) for index in range(rank)]
                reduction_axes = [index for index in range(1, rank + 1) if index != rank - 1]
                broadcast = [1] * rank
                broadcast[-1] = channels
                lines.append(f"        self.{name}_axis = {axis}")
                lines.append(f"        self.{name}_groups = {groups}")
                lines.append(f"        self.{name}_permutation = {permutation!r}")
                lines.append(f"        self.{name}_inverse = {inverse!r}")
                lines.append(f"        self.{name}_reduction_axes = {reduction_axes!r}")
                lines.append(f"        self.{name}_broadcast = {broadcast!r}")
                lines.append(
                    f"        self.{name}_epsilon = {float(params.get('epsilon', 0.001))!r}"
                )
                lines.append(
                    f"        self.{name}_gamma = tf.Variable(_initialize([{channels}], {params.get('gamma_initializer', 'ones')!r}, 'float32'), name={name + '_gamma'!r})"
                    if bool(params.get("scale", True))
                    else f"        self.{name}_gamma = None"
                )
                lines.append(
                    f"        self.{name}_beta = tf.Variable(_initialize([{channels}], {params.get('beta_initializer', 'zeros')!r}, 'float32'), name={name + '_beta'!r})"
                    if bool(params.get("center", True))
                    else f"        self.{name}_beta = None"
                )
            elif definition == "layer.dense":
                input_features = self._static_dimension(source_spec, -1)
                if input_features is None:
                    raise ValueError(
                        f"TensorFlow Dense node {node.id} requires a known input feature dimension"
                    )
                units = int(params.get("units", 32))
                lines.append(
                    f"        self.{name}_kernel = tf.Variable(tf.zeros([{input_features}, {units}], tf.float32), name={name + '_kernel'!r})"
                )
                if bool(params.get("use_bias", True)):
                    lines.append(
                        f"        self.{name}_bias = tf.Variable(tf.zeros([{units}], tf.float32), name={name + '_bias'!r})"
                    )
                else:
                    lines.append(f"        self.{name}_bias = None")
            elif definition == "layer.embedding":
                vocab = int(params.get("vocabulary_size", 10000))
                dim = int(params.get("embedding_dim", 128))
                lines.append(
                    f"        self.{name}_table = tf.Variable(tf.zeros([{vocab}, {dim}], tf.float32), name={name + '_table'!r})"
                )
            elif definition == "layer.batch_norm":
                if source_spec is None:
                    raise ValueError(
                        f"TensorFlow BatchNorm node {node.id} requires a known input rank"
                    )
                rank = source_spec.rank
                axis = self._axis(params.get("axis", -1), rank)
                channels = self._static_dimension(source_spec, axis)
                if channels is None:
                    raise ValueError(
                        f"TensorFlow BatchNorm node {node.id} requires a known normalized dimension"
                    )
                lines.append(f"        self.{name}_axis = {axis}")
                lines.append(
                    f"        self.{name}_epsilon = {float(params.get('epsilon', 0.001))!r}"
                )
                lines.append(
                    f"        self.{name}_gamma = tf.Variable(tf.ones([{channels}], tf.float32), name={name + '_gamma'!r})"
                    if bool(params.get("scale", True))
                    else f"        self.{name}_gamma = None"
                )
                lines.append(
                    f"        self.{name}_beta = tf.Variable(tf.zeros([{channels}], tf.float32), name={name + '_beta'!r})"
                    if bool(params.get("center", True))
                    else f"        self.{name}_beta = None"
                )
            elif definition == "layer.layer_norm":
                if source_spec is None:
                    raise ValueError(
                        f"TensorFlow LayerNorm node {node.id} requires a known input shape"
                    )
                axes = self._normalization_axes(params.get("axis", [-1]), source_spec.rank)
                shape = cast(list[Any], source_spec.shape)
                normalized_shape = [shape[index] for index in axes]
                if not all(isinstance(value, int) for value in normalized_shape):
                    raise ValueError(
                        f"TensorFlow LayerNorm node {node.id} requires known normalized dimensions"
                    )
                broadcast = [1] * source_spec.rank
                for axis, dimension in zip(axes, normalized_shape, strict=True):
                    broadcast[axis] = int(dimension)
                lines.append(f"        self.{name}_axes = {axes!r}")
                lines.append(
                    f"        self.{name}_epsilon = {float(params.get('epsilon', 0.001))!r}"
                )
                lines.append(f"        self.{name}_broadcast = {broadcast!r}")
                lines.append(
                    f"        self.{name}_gamma = tf.Variable(tf.ones({normalized_shape!r}, tf.float32), name={name + '_gamma'!r})"
                    if bool(params.get("scale", True))
                    else f"        self.{name}_gamma = None"
                )
                lines.append(
                    f"        self.{name}_beta = tf.Variable(tf.zeros({normalized_shape!r}, tf.float32), name={name + '_beta'!r})"
                    if bool(params.get("center", True))
                    else f"        self.{name}_beta = None"
                )
            elif definition == "conv.conv3d":
                if source_spec is None or source_spec.rank != 5:
                    raise ValueError(f"TensorFlow Conv3D node {node.id} requires a rank-5 input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = 4 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"TensorFlow Conv3D node {node.id} requires known input channels")
                kernel_size = self._spatial_values(params, "kernel_size", 3, [3, 3, 3])
                filters = int(params.get("filters", 32))
                groups = int(params.get("groups", 1))
                if input_channels % groups != 0 or filters % groups != 0:
                    raise ValueError("TensorFlow Conv3D channels and filters must be divisible by groups")
                lines.append(
                    f"        self.{name}_kernel = tf.Variable(tf.zeros([{kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]}, {input_channels // groups}, {filters}], tf.float32), name={name + '_kernel'!r})"
                )
                lines.append(
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{filters}], tf.float32), name={name + '_bias'!r})"
                    if bool(params.get("use_bias", True))
                    else f"        self.{name}_bias = None"
                )
            elif definition == "conv.depthwise":
                if source_spec is None:
                    raise ValueError(f"TensorFlow depthwise node {node.id} requires a known input shape")
                dimensions = int(params.get("dimensions", 2))
                if dimensions not in {1, 2} or source_spec.rank != dimensions + 2:
                    raise ValueError(f"TensorFlow depthwise node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = dimensions + 1 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"TensorFlow depthwise node {node.id} requires known input channels")
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                depth_multiplier = int(params.get("depth_multiplier", 1))
                kernel_shape = (*kernel_size, input_channels, depth_multiplier)
                lines.append(
                    f"        self.{name}_depthwise_kernel = tf.Variable(tf.zeros({kernel_shape!r}, tf.float32), name={name + '_depthwise_kernel'!r})"
                )
                output_channels = input_channels * depth_multiplier
                lines.append(
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{output_channels}], tf.float32), name={name + '_bias'!r})"
                    if bool(params.get("use_bias", True))
                    else f"        self.{name}_bias = None"
                )
            elif definition == "conv.separable":
                if source_spec is None:
                    raise ValueError(f"TensorFlow separable node {node.id} requires a known input shape")
                dimensions = int(params.get("dimensions", 2))
                if dimensions not in {1, 2} or source_spec.rank != dimensions + 2:
                    raise ValueError(f"TensorFlow separable node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = dimensions + 1 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"TensorFlow separable node {node.id} requires known input channels")
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                depth_multiplier = int(params.get("depth_multiplier", 1))
                filters = int(params.get("filters", 32))
                depth_channels = input_channels * depth_multiplier
                depthwise_shape = (*kernel_size, input_channels, depth_multiplier)
                pointwise_shape = (*([1] * dimensions), depth_channels, filters)
                lines.append(
                    f"        self.{name}_depthwise_kernel = tf.Variable(tf.zeros({depthwise_shape!r}, tf.float32), name={name + '_depthwise_kernel'!r})"
                )
                lines.append(
                    f"        self.{name}_pointwise_kernel = tf.Variable(tf.zeros({pointwise_shape!r}, tf.float32), name={name + '_pointwise_kernel'!r})"
                )
                lines.append(
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{filters}], tf.float32), name={name + '_bias'!r})"
                    if bool(params.get("use_bias", True))
                    else f"        self.{name}_bias = None"
                )
            elif definition == "conv.transposed":
                if source_spec is None:
                    raise ValueError(f"TensorFlow transposed convolution node {node.id} requires a known input shape")
                dimensions = int(params.get("dimensions", 2))
                if dimensions not in {1, 2, 3} or source_spec.rank != dimensions + 2:
                    raise ValueError(f"TensorFlow transposed convolution node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = dimensions + 1 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"TensorFlow transposed convolution node {node.id} requires known input channels")
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                filters = int(params.get("filters", 32))
                lines.append(
                    f"        self.{name}_kernel = tf.Variable(tf.zeros({(*kernel_size, filters, input_channels)!r}, tf.float32), name={name + '_kernel'!r})"
                )
                lines.append(
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{filters}], tf.float32), name={name + '_bias'!r})"
                    if bool(params.get("use_bias", True))
                    else f"        self.{name}_bias = None"
                )
            elif definition == "conv.conv1d":
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"TensorFlow Conv1D node {node.id} requires a rank-3 input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = 2 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(
                        f"TensorFlow Conv1D node {node.id} requires known input channels"
                    )
                kernel_size = params.get("kernel_size", [3])
                if not isinstance(kernel_size, list) or len(kernel_size) != 1:
                    raise ValueError("TensorFlow Conv1D expects one kernel_size value")
                filters = int(params.get("filters", 32))
                groups = int(params.get("groups", 1))
                if input_channels % groups != 0 or filters % groups != 0:
                    raise ValueError(
                        "TensorFlow Conv1D channels and filters must be divisible by groups"
                    )
                kernel = int(kernel_size[0])
                lines.append(
                    f"        self.{name}_kernel = tf.Variable(tf.zeros([{kernel}, {input_channels // groups}, {filters}], tf.float32), name={name + '_kernel'!r})"
                )
                lines.append(
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{filters}], tf.float32), name={name + '_bias'!r})"
                    if bool(params.get("use_bias", True))
                    else f"        self.{name}_bias = None"
                )
            elif definition == "conv.conv2d":
                if source_spec is None or source_spec.rank != 4:
                    raise ValueError(f"TensorFlow Conv2D node {node.id} requires a rank-4 input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = 3 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(
                        f"TensorFlow Conv2D node {node.id} requires known input channels"
                    )
                groups = int(params.get("groups", 1))
                if groups != 1:
                    raise ValueError("Native TensorFlow Conv2D currently supports groups=1 only")
                kernel_size = self._spatial_tuple(params, "kernel_size", [3, 3])
                filters = int(params.get("filters", 32))
                lines.append(
                    f"        self.{name}_kernel = tf.Variable(tf.zeros([{kernel_size[0]}, {kernel_size[1]}, {input_channels}, {filters}], tf.float32), name={name + '_kernel'!r})"
                )
                lines.append(
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{filters}], tf.float32), name={name + '_bias'!r})"
                    if bool(params.get("use_bias", True))
                    else f"        self.{name}_bias = None"
                )
            elif definition in {
                "recurrent.simple_rnn",
                "recurrent.bidirectional",
                "recurrent.stack",
            }:
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"TensorFlow recurrent node {node.id} requires a rank-3 input")
                input_size = self._static_dimension(source_spec, -1)
                if input_size is None:
                    raise ValueError(
                        f"Recurrent node {node.id} requires a known feature dimension"
                    )
                if (
                    params.get("activation", "tanh") != "tanh"
                    or params.get("recurrent_activation", "sigmoid") != "sigmoid"
                ):
                    raise ValueError(
                        "Native TensorFlow recurrent export supports tanh/sigmoid only"
                    )
                if (
                    float(params.get("dropout", 0.0)) != 0.0
                    or float(params.get("recurrent_dropout", 0.0)) != 0.0
                ):
                    raise ValueError("TensorFlow recurrent export does not support recurrent dropout")
                if (
                    params.get("go_backwards", False)
                    or params.get("stateful", False)
                    or params.get("unroll", False)
                ):
                    raise ValueError(
                        "TensorFlow recurrent export does not support go_backwards, stateful, or unroll"
                    )
                units = int(params.get("units", 64))
                cell = (
                    "simple_rnn"
                    if definition == "recurrent.simple_rnn"
                    else str(params.get("cell", "lstm"))
                )
                if cell not in {"simple_rnn", "lstm", "gru"}:
                    raise ValueError(f"Unsupported recurrent cell: {cell}")
                if definition == "recurrent.bidirectional":
                    merge_mode = str(params.get("merge_mode", "concat"))
                    if merge_mode not in {"sum", "mul", "concat", "ave"}:
                        raise ValueError(f"Unsupported bidirectional merge mode: {merge_mode}")
                    if params.get("return_state", False):
                        raise ValueError("TensorFlow bidirectional export does not expose recurrent state ports")
                if definition == "recurrent.stack":
                    if str(params.get("merge_mode", "concat")) != "concat":
                        raise ValueError("Recurrent stack merge_mode must remain concat")
                    if params.get("return_state", False):
                        raise ValueError("TensorFlow recurrent stack does not expose recurrent state ports")
                gate_count = {"simple_rnn": 1, "lstm": 4, "gru": 3}[cell]
                bias = bool(params.get("use_bias", True))
                if definition == "recurrent.simple_rnn":
                    lines.append(
                        f"        self.{name}_kernel = tf.Variable(tf.zeros([{input_size + units}, {units}], tf.float32), name={name + '_kernel'!r})"
                    )
                    lines.append(
                        f"        self.{name}_bias = tf.Variable(tf.zeros([{units}], tf.float32), name={name + '_bias'!r})"
                        if bias
                        else f"        self.{name}_bias = None"
                    )
                elif definition == "recurrent.bidirectional":
                    for direction in ("forward", "backward"):
                        lines.append(
                            f"        self.{name}_{direction}_kernel = tf.Variable(tf.zeros([{input_size + units}, {units * gate_count}], tf.float32), name={name + '_' + direction + '_kernel'!r})"
                        )
                        lines.append(
                            f"        self.{name}_{direction}_bias = tf.Variable(tf.zeros([{units * gate_count}], tf.float32), name={name + '_' + direction + '_bias'!r})"
                            if bias
                            else f"        self.{name}_{direction}_bias = None"
                        )
                else:
                    layers = int(params.get("layers", 2))
                    if layers < 1:
                        raise ValueError("Recurrent stack layers must be positive")
                    layer_input_size = input_size
                    for layer_index in range(layers):
                        lines.append(
                            f"        self.{name}_layer{layer_index}_kernel = tf.Variable(tf.zeros([{layer_input_size + units}, {units * gate_count}], tf.float32), name={name + '_layer' + str(layer_index) + '_kernel'!r})"
                        )
                        lines.append(
                            f"        self.{name}_layer{layer_index}_bias = tf.Variable(tf.zeros([{units * gate_count}], tf.float32), name={name + '_layer' + str(layer_index) + '_bias'!r})"
                            if bias
                            else f"        self.{name}_layer{layer_index}_bias = None"
                        )
                        layer_input_size = units
            elif definition == "transformer.positional_encoding":
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"TensorFlow positional encoding node {node.id} requires a rank-3 input")
                if str(params.get("mode", "sinusoidal")) == "learned":
                    sequence_length = self._static_dimension(source_spec, 1)
                    feature_dimension = self._static_dimension(source_spec, 2)
                    if sequence_length is None or feature_dimension is None:
                        raise ValueError(
                            f"TensorFlow learned positional encoding node {node.id} requires static sequence and feature dimensions"
                        )
                    lines.append(
                        f"        self.{name}_embedding = tf.Variable(tf.zeros([{sequence_length}, {feature_dimension}], tf.float32), trainable={bool(params.get('trainable', True))!r}, name={name + '_embedding'!r})"
                    )
            elif definition in {"recurrent.lstm", "recurrent.gru"}:
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"TensorFlow recurrent node {node.id} requires a rank-3 input")
                input_size = self._static_dimension(source_spec, -1)
                if input_size is None:
                    raise ValueError(
                        f"TensorFlow recurrent node {node.id} requires a known feature dimension"
                    )
                if (
                    params.get("activation", "tanh") != "tanh"
                    or params.get("recurrent_activation", "sigmoid") != "sigmoid"
                ):
                    raise ValueError(
                        "Native TensorFlow recurrent export supports tanh/sigmoid only"
                    )
                units = int(params.get("units", 64))
                gate_count = 4 if definition == "recurrent.lstm" else 3
                lines.append(
                    f"        self.{name}_kernel = tf.Variable(tf.zeros([{input_size + units}, {units * gate_count}], tf.float32), name={name + '_kernel'!r})"
                )
                lines.append(
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{units * gate_count}], tf.float32), name={name + '_bias'!r})"
                    if bool(params.get("use_bias", True))
                    else f"        self.{name}_bias = None"
                )

        jit_compile = bool(
            graph.model_configuration.native.tensorflow.compiler.parameters.get(
                "jit_compile", False
            )
        )
        lines.extend(
            [
                "",
                f"    @tf.function(jit_compile={jit_compile!r})",
                "    def __call__(self, inputs, return_intermediates=False):",
            ]
        )
        variables: dict[str, str] = {}
        port_variables: dict[tuple[str, str], str] = {}
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            definition = item.definition_id
            name = item.variable_name
            params = node.parameters
            source_vars = [
                port_variables[(binding.source_node_id, binding.source_port_id)]
                for binding in item.input_bindings
            ]
            source = source_vars[0] if source_vars else ""
            manifest = catalog.get(definition, node.definition_version)
            if manifest.operation is not None:
                lines.append(
                    f"        {name} = {emit_operation_source(manifest, 'tensorflow', source_vars, params)}"
                )
            elif manifest.source_kind == "runtime_input":
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
            elif definition in {"input.constant", "input.parameter"}:
                lines.append(f"        {name} = self.{name}")
            elif definition == "math.index":
                expression = self._index_expression(params.get("expression", ":"))
                lines.append(f"        {name} = {expression.replace('x', source, 1)}")
            elif definition == "merge.average":
                lines.append(f"        {name} = tf.add_n([{', '.join(source_vars)}]) / {len(source_vars)!r}")
            elif definition == "merge.split":
                lines.append(
                    f"        {name} = tf.split({source}, num_or_size_splits={int(params.get('sections', 2))!r}, axis={int(params.get('axis', -1))!r})"
                )
            elif definition == "output.detection":
                lines.append(f"        {name} = ({', '.join(source_vars)})")
            elif definition == "layer.dense":
                lines.append(f"        {name} = tf.matmul({source}, self.{name}_kernel)")
                lines.append(
                    f"        if self.{name}_bias is not None: {name} = {name} + self.{name}_bias"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = _activation({name}, {activation!r})")
            elif definition == "layer.activation":
                lines.append(
                    f"        {name} = _activation({source}, {str(params.get('activation', 'relu'))!r})"
                )
            elif definition == "layer.dropout":
                lines.append(f"        {name} = {source}")
            elif definition == "layer.flatten":
                lines.append(f"        {name} = tf.reshape({source}, [tf.shape({source})[0], -1])")
            elif definition == "layer.prelu":
                alpha_shape = [1, *[int(value) for value in summaries[item.input_node_ids[0]].shape[1:]]]
                shared_axes = params.get("shared_axes")
                if isinstance(shared_axes, list):
                    for shared_axis in shared_axes:
                        alpha_shape[int(shared_axis)] = 1
                alpha = f"tf.reshape(self.{name}_alpha, {alpha_shape!r})"
                lines.append(f"        {name} = tf.where({source} >= 0, {source}, {source} * {alpha})")
            elif definition == "layer.rms_norm":
                lines.append(
                    f"        {name} = {source} / tf.sqrt(tf.reduce_mean(tf.square({source}), axis=self.{name}_axis, keepdims=True) + self.{name}_epsilon)"
                )
                lines.append(
                    f"        {name} = {name} * tf.reshape(self.{name}_scale, self.{name}_broadcast)"
                )
            elif definition == "layer.unit_norm":
                lines.append(f"        {name} = tf.math.l2_normalize({source}, axis=self.{name}_axis)")
            elif definition == "layer.group_norm":
                lines.append(f"        {name}_current = tf.transpose({source}, self.{name}_permutation)")
                lines.append(
                    f"        {name}_grouped = tf.reshape({name}_current, tf.concat([tf.shape({name}_current)[:-1], [self.{name}_groups, tf.shape({name}_current)[-1] // self.{name}_groups]], axis=0))"
                )
                lines.append(
                    f"        {name}_mean, {name}_variance = tf.nn.moments({name}_grouped, axes=self.{name}_reduction_axes, keepdims=True)"
                )
                lines.append(
                    f"        {name}_current = tf.reshape(({name}_grouped - {name}_mean) / tf.sqrt({name}_variance + self.{name}_epsilon), tf.shape({name}_current))"
                )
                lines.append(
                    f"        if self.{name}_gamma is not None: {name}_current = {name}_current * tf.reshape(self.{name}_gamma, self.{name}_broadcast)"
                )
                lines.append(
                    f"        if self.{name}_beta is not None: {name}_current = {name}_current + tf.reshape(self.{name}_beta, self.{name}_broadcast)"
                )
                lines.append(f"        {name} = tf.transpose({name}_current, self.{name}_inverse)")
            elif definition == "layer.reshape":
                target_shape = [int(value) for value in params.get("target_shape", [-1])]
                lines.append(
                    f"        {name} = tf.reshape({source}, [tf.shape({source})[0]] + {target_shape!r})"
                )
            elif definition == "layer.permute":
                axes = [int(value) for value in params.get("axes", [2, 1])]
                lines.append(f"        {name} = tf.transpose({source}, [0] + {axes!r})")
            elif definition == "layer.repeat":
                repeats = int(params.get("repeats", 2))
                axis = int(params.get("axis", 1))
                if axis == 1:
                    lines.append(
                        f"        {name} = tf.repeat(tf.expand_dims({source}, axis=1), {repeats!r}, axis=1)"
                    )
                else:
                    lines.append(
                        f"        {name} = tf.repeat({source}, {repeats!r}, axis={axis!r})"
                    )
            elif definition == "layer.embedding":
                lines.append(
                    f"        {name} = tf.gather(self.{name}_table, tf.cast({source}, tf.int32))"
                )
            elif definition == "layer.batch_norm":
                rank = summaries[item.input_node_ids[0]].rank
                axis = self._axis(params.get("axis", -1), rank)
                channels = self._static_dimension(summaries[item.input_node_ids[0]], axis)
                assert channels is not None
                broadcast = [1] * rank
                broadcast[axis] = channels
                lines.append(
                    f"        {name} = {source} / tf.sqrt(tf.cast(1.0 + self.{name}_epsilon, {source}.dtype))"
                )
                lines.append(
                    f"        if self.{name}_gamma is not None: {name} = {name} * tf.reshape(self.{name}_gamma, {broadcast!r})"
                )
                lines.append(
                    f"        if self.{name}_beta is not None: {name} = {name} + tf.reshape(self.{name}_beta, {broadcast!r})"
                )
            elif definition == "layer.layer_norm":
                lines.append(
                    f"        {name}_mean, {name}_variance = tf.nn.moments({source}, axes=self.{name}_axes, keepdims=True)"
                )
                lines.append(
                    f"        {name} = tf.nn.batch_normalization({source}, {name}_mean, {name}_variance, self.{name}_beta, self.{name}_gamma, self.{name}_epsilon)"
                )
            elif definition == "conv.conv1d":
                kernel_size = int(params.get("kernel_size", [3])[0])
                strides = int(params.get("strides", [1])[0])
                dilation = int(params.get("dilation_rate", [1])[0])
                padding = str(params.get("padding", "same"))
                data_format = str(params.get("data_format", "channels_last"))
                groups = int(params.get("groups", 1))
                lines.append(
                    f"        {name} = _conv1d({source}, self.{name}_kernel, self.{name}_bias, {strides!r}, {dilation!r}, {padding!r}, {data_format!r}, {groups!r})"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = _activation({name}, {activation!r})")
            elif definition == "conv.conv2d":
                kernel_size = self._spatial_tuple(params, "kernel_size", [3, 3])
                strides = self._spatial_tuple(params, "strides", [1, 1])
                dilation = self._spatial_tuple(params, "dilation_rate", [1, 1])
                padding = str(params.get("padding", "same"))
                data_format = str(params.get("data_format", "channels_last"))
                lines.append(
                    f"        {name} = _conv2d({source}, self.{name}_kernel, self.{name}_bias, {strides!r}, {dilation!r}, {padding!r}, {data_format!r})"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = _activation({name}, {activation!r})")
            elif definition == "conv.conv3d":
                strides = self._spatial_values(params, "strides", 3, [1, 1, 1])
                dilation = self._spatial_values(params, "dilation_rate", 3, [1, 1, 1])
                padding = str(params.get("padding", "same"))
                data_format = str(params.get("data_format", "channels_last"))
                groups = int(params.get("groups", 1))
                lines.append(
                    f"        {name} = _conv3d({source}, self.{name}_kernel, self.{name}_bias, {strides!r}, {dilation!r}, {padding!r}, {data_format!r}, {groups!r})"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = _activation({name}, {activation!r})")
            elif definition == "conv.depthwise":
                dimensions = int(params.get("dimensions", 2))
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                padding = str(params.get("padding", "same"))
                data_format = str(params.get("data_format", "channels_last"))
                lines.append(
                    f"        {name} = _depthwise({source}, self.{name}_depthwise_kernel, self.{name}_bias, {strides!r}, {dilation!r}, {padding!r}, {data_format!r}, {dimensions!r})"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = _activation({name}, {activation!r})")
            elif definition == "conv.separable":
                dimensions = int(params.get("dimensions", 2))
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                padding = str(params.get("padding", "same"))
                data_format = str(params.get("data_format", "channels_last"))
                lines.append(
                    f"        {name} = _separable({source}, self.{name}_depthwise_kernel, self.{name}_pointwise_kernel, self.{name}_bias, {strides!r}, {dilation!r}, {padding!r}, {data_format!r}, {dimensions!r})"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = _activation({name}, {activation!r})")
            elif definition == "conv.transposed":
                dimensions = int(params.get("dimensions", 2))
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                padding = str(params.get("padding", "valid"))
                data_format = str(params.get("data_format", "channels_last"))
                output_padding_value = params.get("output_padding")
                output_padding = (
                    None
                    if output_padding_value is None
                    else self._spatial_nonnegative(params, "output_padding", dimensions, [0] * dimensions)
                )
                filters = int(params.get("filters", 32))
                lines.append(
                    f"        {name} = _transposed({source}, self.{name}_kernel, self.{name}_bias, {filters!r}, {strides!r}, {dilation!r}, {padding!r}, {output_padding!r}, {data_format!r}, {dimensions!r})"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = _activation({name}, {activation!r})")
            elif definition in {
                "conv.max_pool1d",
                "conv.max_pool2d",
                "conv.max_pool3d",
                "conv.average_pool1d",
                "conv.average_pool2d",
                "conv.average_pool3d",
            }:
                dimensions = int(definition[-2])
                pool_size = self._spatial_values(params, "pool_size", dimensions, [2] * dimensions)
                strides_value = params.get("strides")
                strides = (
                    pool_size
                    if strides_value is None
                    else self._spatial_values(params, "strides", dimensions, list(pool_size))
                )
                padding = str(params.get("padding", "valid"))
                data_format = str(params.get("data_format", "channels_last"))
                mode = "max" if definition.startswith("conv.max") else "avg"
                lines.append(
                    f"        {name} = _pool_nd({source}, {pool_size!r}, {strides!r}, {padding!r}, {data_format!r}, {mode!r})"
                )
            elif definition in {
                "conv.global_pool1d",
                "conv.global_pool2d",
                "conv.global_pool3d",
            }:
                dimensions = int(definition[-2])
                mode = str(params.get("mode", "average"))
                data_format = str(params.get("data_format", "channels_last"))
                keepdims = bool(params.get("keepdims", False))
                lines.append(
                    f"        {name} = _global_pool_nd({source}, {dimensions!r}, {data_format!r}, {mode!r}, {keepdims!r})"
                )
            elif definition == "conv.upsampling":
                dimensions = int(params.get("dimensions", 2))
                size = self._spatial_values(params, "size", dimensions, [2] * dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                interpolation = str(params.get("interpolation", "nearest"))
                lines.append(
                    f"        {name} = _upsample_nd({source}, {size!r}, {data_format!r}, {interpolation!r})"
                )
            elif definition == "conv.padding":
                dimensions = int(params.get("dimensions", 2))
                pairs = self._spatial_pairs(params, "padding", dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                lines.append(f"        {name} = _pad_nd({source}, {pairs!r}, {data_format!r})")
            elif definition == "conv.cropping":
                dimensions = int(params.get("dimensions", 2))
                pairs = self._spatial_pairs(params, "cropping", dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                rank = dimensions + 2
                spatial_indices = (
                    list(range(1, dimensions + 1))
                    if data_format == "channels_last"
                    else list(range(2, dimensions + 2))
                )
                slices = ["slice(None)"] * rank
                for index, (before, after) in zip(spatial_indices, pairs, strict=True):
                    slices[index] = f"slice({before}, {-after if after else 'None'})"
                lines.append(f"        {name} = {source}[({', '.join(slices)},)]")
            elif definition == "recurrent.simple_rnn":
                return_sequences = bool(params.get("return_sequences", False))
                return_state = bool(params.get("return_state", False))
                if return_state and return_sequences:
                    sequence_name = f"{name}_sequence"
                    lines.append(
                        f"        {sequence_name} = _simple_rnn({source}, self.{name}_kernel, self.{name}_bias, {int(params.get('units', 64))}, True)"
                    )
                    lines.append(
                        f"        {name} = ({sequence_name}, {sequence_name}[:, -1, :])"
                    )
                else:
                    lines.append(
                        f"        {name} = _simple_rnn({source}, self.{name}_kernel, self.{name}_bias, {int(params.get('units', 64))}, {return_sequences!r})"
                    )
                    if return_state:
                        lines.append(f"        {name} = ({name}, {name})")
            elif definition == "recurrent.bidirectional":
                lines.append(
                    f"        {name} = _bidirectional({source}, {str(params.get('cell', 'lstm'))!r}, self.{name}_forward_kernel, self.{name}_forward_bias, self.{name}_backward_kernel, self.{name}_backward_bias, {int(params.get('units', 64))}, {str(params.get('merge_mode', 'concat'))!r}, {bool(params.get('return_sequences', False))!r})"
                )
            elif definition == "recurrent.stack":
                cell = str(params.get("cell", "lstm"))
                layers = int(params.get("layers", 2))
                current = source
                for layer_index in range(layers):
                    layer_name = f"{name}_layer{layer_index}"
                    layer_return_sequences = layer_index < layers - 1 or bool(
                        params.get("return_sequences", False)
                    )
                    lines.append(
                        f"        {layer_name} = _recurrent({current}, {cell!r}, self.{layer_name}_kernel, self.{layer_name}_bias, {int(params.get('units', 64))}, {layer_return_sequences!r})"
                    )
                    current = layer_name
                lines.append(f"        {name} = {current}")
            elif definition == "transformer.attention_mask":
                lines.append(f"        {name} = {source}")
            elif definition == "transformer.residual":
                lines.append(f"        {name} = tf.add_n([{', '.join(source_vars)}])")
            elif definition == "transformer.positional_encoding":
                if str(params.get("mode", "sinusoidal")) == "learned":
                    lines.append(f"        {name} = {source} + self.{name}_embedding[None, :, :]" )
                else:
                    lines.append(f"        {name} = _sinusoidal_encoding({source})")
            elif definition == "recurrent.lstm":
                lines.append(
                    f"        {name} = _lstm({source}, self.{name}_kernel, self.{name}_bias, {int(params.get('units', 64))}, {bool(params.get('return_sequences', False))!r})"
                )
            elif definition == "recurrent.gru":
                lines.append(
                    f"        {name} = _gru({source}, self.{name}_kernel, self.{name}_bias, {int(params.get('units', 64))}, {bool(params.get('return_sequences', False))!r})"
                )
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(f"        {name} = tf.add_n([{', '.join(source_vars)}])")
            elif definition == "merge.concatenate":
                lines.append(
                    f"        {name} = tf.concat([{', '.join(source_vars)}], axis={int(params.get('axis', -1))!r})"
                )
            elif manifest.semantic_role == "output_adapter":
                lines.append(f"        {name} = {source}")
            else:
                raise ValueError(f"No native TensorFlow adapter registered for {definition}")
            output_ports = [
                port
                for port in materialized_outputs(manifest, params)
                if self._condition_matches(port.enabled_when, params)
            ]
            if len(output_ports) > 1:
                for index, port in enumerate(output_ports):
                    port_variables[(item.node_id, port.id)] = f"{name}[{index}]"
                variables[item.node_id] = f"{name}[0]"
            else:
                port_variables[(item.node_id, output_ports[0].id)] = name
                variables[item.node_id] = name
        outputs = [port_variables[(binding.node_id, binding.port_id)] for binding in ir.output_bindings]
        result = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
        trace_entries: list[str] = []
        for trace_item in ir.ordered_nodes:
            trace_manifest = catalog.get(
                trace_item.definition_id,
                nodes[trace_item.node_id].definition_version,
            )
            trace_ports = ", ".join(
                f"{port.id!r}: {port_variables[(trace_item.node_id, port.id)]}"
                for port in materialized_outputs(
                    trace_manifest,
                    nodes[trace_item.node_id].parameters,
                )
                if (trace_item.node_id, port.id) in port_variables
            )
            trace_entries.append(
                f"{trace_item.node_id!r}: {{'outputs': {{{trace_ports}}}}}"
            )
        trace = "{" + ", ".join(trace_entries) + "}"
        lines.append(f"        if return_intermediates: return {result}, {trace}")
        lines.append(f"        return {result}")
        lines.extend(
            ["", "", "def build_model() -> GeneratedModel:", "    return GeneratedModel()", ""]
        )
        return "\n".join(lines)
