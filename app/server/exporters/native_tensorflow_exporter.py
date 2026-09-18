# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from typing import Any, cast

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import CompilationCatalog
from domain.graph import ModelGraph
from domain.tensor import TensorSpec
from exporters.base import SourceExporter


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
    def _axis(value: object, rank: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("TensorFlow normalization expects an integer axis")
        axis = value if value >= 0 else rank + value
        if axis < 0 or axis >= rank:
            raise ValueError(f"Normalization axis {value} is outside rank {rank}")
        return axis

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
    @staticmethod
    def _condition_matches(condition: dict[str, Any], parameters: dict[str, Any]) -> bool:
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
            elif isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

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
            if definition == "layer.group_norm":
                if source_spec is None:
                    raise ValueError(f"TensorFlow GroupNorm node {node.id} requires an inferred input shape")
                axis = self._axis(params.get("axis", -1), source_spec.rank)
                channels = self._static_dimension(source_spec, axis)
                groups = int(params.get("groups", 32))
                if channels is None or groups <= 0 or channels % groups != 0:
                    raise ValueError(f"TensorFlow GroupNorm node {node.id} requires a known channel dimension divisible by groups")
                lines.append(f"        self.{name}_axis = {axis!r}")
                lines.append(f"        self.{name}_groups = {groups!r}")
                lines.append(f"        self.{name}_epsilon = {float(params.get('epsilon', 0.001))!r}")
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
            elif definition == "layer.rms_norm":
                if source_spec is None:
                    raise ValueError(f"TensorFlow RMSNorm node {node.id} requires an inferred input shape")
                axis = self._axis(params.get("axis", -1), source_spec.rank)
                dimension = self._static_dimension(source_spec, axis)
                if dimension is None:
                    raise ValueError(f"TensorFlow RMSNorm node {node.id} requires a known normalized dimension")
                lines.append(f"        self.{name}_axis = {axis!r}")
                lines.append(f"        self.{name}_epsilon = {float(params.get('epsilon', 1e-6))!r}")
                lines.append(f"        self.{name}_gamma = tf.Variable(tf.ones([{dimension}], tf.float32), name={name + '_gamma'!r})")
            elif definition == "layer.prelu":
                if source_spec is None:
                    raise ValueError(f"TensorFlow PReLU node {node.id} requires an inferred input shape")
                shape = cast(list[Any], source_spec.shape)
                shared_axes = params.get("shared_axes")
                shared = set(cast(list[int], shared_axes)) if isinstance(shared_axes, list) else set()
                alpha_shape: list[int] = []
                for axis_index, dimension in enumerate(shape[1:], start=1):
                    if axis_index in shared:
                        alpha_shape.append(1)
                    elif isinstance(dimension, int):
                        alpha_shape.append(dimension)
                    else:
                        raise ValueError(f"TensorFlow PReLU node {node.id} requires known non-shared dimensions")
                lines.append(f"        self.{name}_alpha = tf.Variable(tf.zeros({alpha_shape!r}, tf.float32), name={name + '_alpha'!r})")
            elif definition == "input.parameter":
                source_shape = params.get("shape", [1])
                if not isinstance(source_shape, list):
                    raise ValueError(f"TensorFlow parameter node {node.id} requires a list shape")
                initializer = str(params.get("initializer", "zeros"))
                dtype = str(params.get("dtype", "float32"))
                if initializer == "zeros":
                    initial_source = f"tf.zeros({source_shape!r}, dtype=tf.{dtype})"
                elif initializer == "ones":
                    initial_source = f"tf.ones({source_shape!r}, dtype=tf.{dtype})"
                elif initializer == "random_normal":
                    initial_source = f"tf.random.normal({source_shape!r}, dtype=tf.{dtype})"
                else:
                    raise ValueError(f"Unsupported TensorFlow parameter initializer: {initializer}")
                lines.append(
                    f"        self.{name}_parameter = tf.Variable({initial_source}, trainable={bool(params.get('trainable', True))!r}, name={name + '_parameter'!r})"
                )
            elif definition == "layer.reshape":
                target_shape = tuple(params.get("target_shape", [-1]))
                lines.append(f"        {name} = tf.reshape({source}, [tf.shape({source})[0], *{target_shape!r}])")
            elif definition == "layer.permute":
                axes = tuple(int(value) for value in params.get("axes", [2, 1]))
                lines.append(f"        {name} = tf.transpose({source}, perm={(0, *axes)!r})")
            elif definition == "layer.repeat":
                repeats = int(params.get("repeats", 2))
                axis = int(params.get("axis", 1))
                if axis == 1 and source_spec is not None and source_spec.rank == 2:
                    lines.append(
                        f"        {name} = tf.repeat(tf.expand_dims({source}, axis=1), repeats={repeats!r}, axis=1)"
                    )
                else:
                    lines.append(
                        f"        {name} = tf.repeat({source}, repeats={repeats!r}, axis={axis!r})"
                    )
            elif definition == "layer.unit_norm":
                axis = int(params.get("axis", -1))
                lines.append(f"        {name} = tf.math.l2_normalize({source}, axis={axis!r}, epsilon=1e-12)")
            elif definition == "layer.rms_norm":
                axis = int(params.get("axis", -1))
                lines.append(f"        {name} = {source} * tf.math.rsqrt(tf.reduce_mean(tf.square({source}), axis={axis!r}, keepdims=True) + self.{name}_epsilon)")
                lines.append(f"        {name}_shape = [1] * {name}.shape.rank")
                lines.append(f"        {name}_shape[{axis!r}] = self.{name}_gamma.shape[0]")
                lines.append(f"        {name} = {name} * tf.reshape(self.{name}_gamma, {name}_shape)")
            elif definition == "layer.prelu":
                lines.append(f"        {name} = tf.where({source} >= 0, {source}, self.{name}_alpha * {source})")
            elif definition == "layer.group_norm":
                if source_spec is None:
                    raise ValueError(f"TensorFlow GroupNorm node {node.id} requires an inferred input shape")
                rank = source_spec.rank
                axis = self._axis(params.get("axis", -1), rank)
                channels = self._static_dimension(source_spec, axis)
                if channels is None:
                    raise ValueError(f"TensorFlow GroupNorm node {node.id} requires known channels")
                if axis != rank - 1:
                    permutation = tuple(index for index in range(rank) if index != axis) + (axis,)
                    inverse = tuple(permutation.index(index) for index in range(rank))
                    lines.append(f"        {name}_input = tf.transpose({source}, {permutation!r})")
                else:
                    lines.append(f"        {name}_input = {source}")
                lines.append(f"        {name}_shape = tf.shape({name}_input)")
                lines.append(f"        {name}_grouped = tf.reshape({name}_input, tf.concat([{name}_shape[:-1], [{int(params.get('groups', 32))!r}, {channels // int(params.get('groups', 32))!r}]], axis=0))")
                lines.append(f"        {name}_axes = list(range(1, {rank - 1!r})) + [{rank!r}]")
                lines.append(f"        {name}_mean, {name}_variance = tf.nn.moments({name}_grouped, axes={name}_axes, keepdims=True)")
                lines.append(f"        {name}_grouped = ({name}_grouped - {name}_mean) * tf.math.rsqrt({name}_variance + self.{name}_epsilon)")
                lines.append(f"        {name} = tf.reshape({name}_grouped, {name}_shape)")
                lines.append(f"        if self.{name}_gamma is not None: {name} = {name} * self.{name}_gamma")
                lines.append(f"        if self.{name}_beta is not None: {name} = {name} + self.{name}_beta")
                if axis != rank - 1:
                    lines.append(f"        {name} = tf.transpose({name}, {inverse!r})")
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

        jit_compile = graph.model_configuration.native.tensorflow.jit_compile
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
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None
            manifest = catalog.get(definition, node.definition_version)
            if definition in {"input.constant", "input.parameter"}:
                if not ir.input_node_ids:
                    raise ValueError(
                        f"Generated source node {node.id} requires at least one runtime model input for batch materialization"
                    )
                anchor = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{ir.input_node_ids[0]!r}]"
                source_shape = params.get("shape", [1])
                if not isinstance(source_shape, list):
                    raise ValueError(f"Generated source node {node.id} requires a list shape")
                if definition == "input.constant":
                    dtype = str(params.get("dtype", "float32"))
                    lines.append(
                        f"        {name}_source = tf.fill({source_shape!r}, tf.cast({params.get('value', 0)!r}, tf.{dtype}))"
                    )
                    lines.append(
                        f"        {name} = tf.repeat(tf.expand_dims({name}_source, 0), repeats=tf.shape({anchor})[0], axis=0)"
                    )
                else:
                    lines.append(
                        f"        {name} = tf.repeat(tf.expand_dims(self.{name}_parameter, 0), repeats=tf.shape({anchor})[0], axis=0)"
                    )
            elif manifest.operation is not None:
                lines.append(
                    f"        {name} = {emit_operation_source(manifest, 'tensorflow', source_vars, params)}"
                )
            elif manifest.semantic_role == "input":
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
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
            elif definition == "conv.max_pool2d":
                pool_size = self._spatial_tuple(params, "pool_size", [2, 2])
                strides_value = params.get("strides")
                strides = (
                    pool_size
                    if strides_value is None
                    else self._spatial_tuple(params, "strides", [2, 2])
                )
                padding = str(params.get("padding", "valid"))
                data_format = str(params.get("data_format", "channels_last"))
                lines.append(
                    f"        {name} = _max_pool2d({source}, {pool_size!r}, {strides!r}, {padding!r}, {data_format!r})"
                )
            elif definition == "recurrent.lstm":
                lines.append(
                    f"        {name} = _lstm({source}, self.{name}_kernel, self.{name}_bias, {int(params.get('units', 64))}, {bool(params.get('return_sequences', False))!r})"
                )
            elif definition == "recurrent.gru":
                lines.append(
                    f"        {name} = _gru({source}, self.{name}_kernel, self.{name}_bias, {int(params.get('units', 64))}, {bool(params.get('return_sequences', False))!r})"
                )
            elif definition in {"merge.add", "merge.residual", "merge.skip", "transformer.residual"}:
                lines.append(f"        {name} = tf.add_n([{', '.join(source_vars)}])")
            elif definition == "merge.concatenate":
                lines.append(
                    f"        {name} = tf.concat([{', '.join(source_vars)}], axis={int(params.get('axis', -1))!r})"
                )
            elif definition == "merge.split":
                lines.append(
                    f"        {name} = tf.split({source}, num_or_size_splits={int(params.get('sections', 2))!r}, axis={int(params.get('axis', -1))!r})"
                )
            elif definition == "transformer.attention_mask":
                lines.append(f"        {name} = {source}")
            elif definition == "output.detection":
                lines.append(f"        {name} = ({', '.join(source_vars)})")
            elif manifest.semantic_role == "output_adapter":
                lines.append(f"        {name} = {source}")
            else:
                raise ValueError(f"No native TensorFlow adapter registered for {definition}")
            output_ports = [
                port
                for port in manifest.outputs
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
        trace = "{" + ", ".join(
            f"{item.node_id!r}: {{'output': {variables[item.node_id]}}}"
            for item in ir.ordered_nodes
        ) + "}"
        lines.append(f"        if return_intermediates: return {result}, {trace}")
        lines.append(f"        return {result}")
        lines.extend(
            ["", "", "def build_model() -> GeneratedModel:", "    return GeneratedModel()", ""]
        )
        return "\n".join(lines)
