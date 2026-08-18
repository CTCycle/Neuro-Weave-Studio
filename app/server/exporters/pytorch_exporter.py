from __future__ import annotations

from typing import Any, cast

from domain.compilation import CanonicalModelIR
from domain.graph import ModelGraph
from domain.tensor import TensorSpec
from exporters.base import SourceExporter

###############################################################################
class PyTorchSourceExporter(SourceExporter):

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
        if not isinstance(value, list):
            raise ValueError(f"PyTorch Conv2D expects {key} to contain two values")
        values = cast(list[Any], value)
        if len(values) != 2:
            raise ValueError(f"PyTorch Conv2D expects {key} to contain two values")
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in values):
            raise ValueError(f"PyTorch Conv2D expects positive integer {key} values")
        return int(values[0]), int(values[1])

    # -------------------------------------------------------------------------
    @staticmethod
    def _axis(value: object, rank: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("PyTorch normalization expects an integer axis")
        axis = value if value >= 0 else rank + value
        if axis < 0 or axis >= rank:
            raise ValueError(f"Normalization axis {value} is outside rank {rank}")
        return axis

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalization_axes(value: object, rank: int) -> list[int]:
        if not isinstance(value, list) or not value:
            raise ValueError("PyTorch LayerNorm expects a non-empty axis list")
        values = cast(list[Any], value)
        axes = [PyTorchSourceExporter._axis(item, rank) for item in values]
        if axes != list(range(rank - len(axes), rank)):
            raise ValueError("PyTorch LayerNorm requires normalized axes at the end of the tensor")
        return axes

    # -------------------------------------------------------------------------
    @staticmethod
    def _activation(value: str) -> str:
        try:
            return {
                "linear": "lambda value: value",
                "relu": "torch.relu",
                "sigmoid": "torch.sigmoid",
                "tanh": "torch.tanh",
                "gelu": "torch.nn.functional.gelu",
                "softmax": "lambda value: torch.softmax(value, dim=-1)",
            }[value]
        except KeyError as exc:
            raise ValueError(f"No PyTorch activation adapter registered for {value}") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _channel_permutation(rank: int) -> tuple[int, ...]:
        if rank == 3:
            return (0, 2, 1)
        if rank == 4:
            return (0, 3, 1, 2)
        if rank == 5:
            return (0, 4, 1, 2, 3)
        raise ValueError(f"PyTorch channel-last normalization does not support rank {rank}")

    # -------------------------------------------------------------------------
    def generate(self, graph: ModelGraph, ir: CanonicalModelIR) -> str:
        nodes = {node.id: node for node in graph.nodes}
        summaries = self._summary_map(ir)
        lines = [
            "from __future__ import annotations",
            "",
            "import torch",
            "import torch.nn.functional as F",
            "from torch import Tensor, nn",
            "",
            "",
            "def _same_pad_2d(value: Tensor, kernel_size: tuple[int, int], stride: tuple[int, int], dilation: tuple[int, int], fill_value: float = 0.0) -> Tensor:",
            "    height, width = value.shape[-2], value.shape[-1]",
            "    output_height = (height + stride[0] - 1) // stride[0]",
            "    output_width = (width + stride[1] - 1) // stride[1]",
            "    effective_height = dilation[0] * (kernel_size[0] - 1) + 1",
            "    effective_width = dilation[1] * (kernel_size[1] - 1) + 1",
            "    total_height = max((output_height - 1) * stride[0] + effective_height - height, 0)",
            "    total_width = max((output_width - 1) * stride[1] + effective_width - width, 0)",
            "    top = total_height // 2",
            "    left = total_width // 2",
            "    return F.pad(value, (left, total_width - left, top, total_height - top), value=fill_value)",
            "",
            "",
            "def _set_lstm_forget_bias(layer: nn.LSTM) -> None:",
            "    with torch.no_grad():",
            "        hidden_size = layer.hidden_size",
            "        layer.bias_ih_l0[hidden_size:2 * hidden_size].fill_(1.0)",
            "",
            "",
            "class GeneratedModel(nn.Module):",
            "    def __init__(self) -> None:",
            "        super().__init__()",
        ]

        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            params = node.parameters
            definition = item.definition_id
            name = item.variable_name
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None

            if definition == "layer.dense":
                input_features = self._static_dimension(source_spec, -1)
                units = int(params.get("units", 32))
                bias = bool(params.get("use_bias", True))
                if input_features is None:
                    lines.append(f"        self.{name} = nn.LazyLinear({units!r}, bias={bias!r})")
                else:
                    lines.append(
                        f"        self.{name} = nn.Linear({input_features!r}, {units!r}, bias={bias!r})"
                    )
            elif definition == "layer.dropout":
                lines.append(f"        self.{name} = nn.Dropout({float(params.get('rate', 0.5))!r})")
            elif definition == "layer.flatten":
                lines.append(f"        self.{name} = nn.Flatten(start_dim=1)")
            elif definition == "layer.batch_norm":
                if source_spec is None:
                    raise ValueError(f"Cannot infer the rank for batch normalization node {node.id}")
                rank = source_spec.rank
                axis = self._axis(params.get("axis", -1), rank)
                if axis not in {1, rank - 1}:
                    raise ValueError("PyTorch BatchNorm supports channel axis 1 or the last axis")
                channel_count = self._static_dimension(source_spec, axis)
                momentum = 1.0 - float(params.get("momentum", 0.99))
                epsilon = float(params.get("epsilon", 0.001))
                center = bool(params.get("center", True))
                scale = bool(params.get("scale", True))
                if center != scale:
                    raise ValueError("PyTorch BatchNorm requires center and scale to match")
                batch_norm_class = "BatchNorm1d" if rank in {2, 3} else f"BatchNorm{rank - 2}d"
                if channel_count is None:
                    lines.append(
                        f"        self.{name} = nn.Lazy{batch_norm_class}({epsilon!r}, {momentum!r}, affine={center!r})"
                    )
                else:
                    lines.append(
                        f"        self.{name} = nn.{batch_norm_class}({channel_count!r}, eps={epsilon!r}, momentum={momentum!r}, affine={center!r})"
                    )
            elif definition == "layer.layer_norm":
                if source_spec is None:
                    raise ValueError(f"Cannot infer the shape for layer normalization node {node.id}")
                axes = self._normalization_axes(params.get("axis", [-1]), source_spec.rank)
                shape = cast(list[Any], source_spec.shape)
                normalized_shape = [shape[index] for index in axes]
                if not all(isinstance(item, int) for item in normalized_shape):
                    raise ValueError(f"LayerNorm node {node.id} requires known normalized dimensions")
                if not bool(params.get("center", True)) or not bool(params.get("scale", True)):
                    raise ValueError("PyTorch LayerNorm requires center and scale")
                lines.append(
                    f"        self.{name} = nn.LayerNorm({normalized_shape!r}, eps={float(params.get('epsilon', 0.001))!r})"
                )
            elif definition == "layer.embedding":
                lines.append(
                    f"        self.{name} = nn.Embedding({int(params.get('vocabulary_size', 10000))!r}, {int(params.get('embedding_dim', 128))!r})"
                )
            elif definition == "conv.conv2d":
                if source_spec is None or source_spec.rank != 4:
                    raise ValueError(f"PyTorch Conv2D node {node.id} requires a rank-4 input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = 3 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                kernel_size = self._spatial_tuple(params, "kernel_size", [3, 3])
                strides = self._spatial_tuple(params, "strides", [1, 1])
                dilation = self._spatial_tuple(params, "dilation_rate", [1, 1])
                filters = int(params.get("filters", 32))
                groups = int(params.get("groups", 1))
                bias = bool(params.get("use_bias", True))
                if input_channels is None and groups != 1:
                    raise ValueError("PyTorch LazyConv2d cannot infer grouped input channels")
                module = "nn.Conv2d" if input_channels is not None else "nn.LazyConv2d"
                constructor = (
                    f"{input_channels!r}, {filters!r}"
                    if input_channels is not None
                    else f"{filters!r}"
                )
                lines.append(
                    f"        self.{name} = {module}({constructor}, kernel_size={kernel_size!r}, stride={strides!r}, padding=0, dilation={dilation!r}, groups={groups!r}, bias={bias!r})"
                )
            elif definition == "conv.max_pool2d":
                pool_size = self._spatial_tuple(params, "pool_size", [2, 2])
                strides_value = params.get("strides")
                strides = (
                    pool_size
                    if strides_value is None
                    else self._spatial_tuple(params, "strides", [2, 2])
                )
                lines.append(
                    f"        self.{name} = nn.MaxPool2d(kernel_size={pool_size!r}, stride={strides!r}, padding=0)"
                )
            elif definition in {"recurrent.lstm", "recurrent.gru"}:
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"PyTorch recurrent node {node.id} requires a rank-3 input")
                input_size = self._static_dimension(source_spec, -1)
                if input_size is None:
                    raise ValueError(f"Recurrent node {node.id} requires a known feature dimension")
                if params.get("activation", "tanh") != "tanh" or params.get("recurrent_activation", "sigmoid") != "sigmoid":
                    raise ValueError("PyTorch recurrent export supports the canonical tanh/sigmoid activations only")
                if float(params.get("dropout", 0.0)) != 0.0 or float(params.get("recurrent_dropout", 0.0)) != 0.0:
                    raise ValueError("PyTorch recurrent export does not support recurrent dropout")
                if params.get("go_backwards", False) or params.get("stateful", False) or params.get("unroll", False):
                    raise ValueError("PyTorch recurrent export does not support go_backwards, stateful, or unroll")
                units = int(params.get("units", 64))
                recurrent_class = "LSTM" if definition == "recurrent.lstm" else "GRU"
                lines.append(
                    f"        self.{name} = nn.{recurrent_class}({input_size!r}, {units!r}, batch_first=True, bias={bool(params.get('use_bias', True))!r})"
                )
                if definition == "recurrent.lstm" and bool(params.get("unit_forget_bias", True)):
                    lines.append(f"        _set_lstm_forget_bias(self.{name})")

        lines.extend(
            [
                "",
                "    def forward(self, inputs: Tensor | dict[str, Tensor]) -> Tensor | tuple[Tensor, ...]:",
            ]
        )
        variables: dict[str, str] = {}
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            definition = item.definition_id
            name = item.variable_name
            source_vars = [variables[source_id] for source_id in item.input_node_ids]
            source = source_vars[0] if source_vars else ""
            params = node.parameters
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None

            if definition.startswith("input."):
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
                variables[item.node_id] = name
            elif definition in {
                "layer.dense",
                "layer.dropout",
                "layer.flatten",
                "layer.batch_norm",
                "layer.layer_norm",
            }:
                if definition == "layer.batch_norm" and source_spec is not None:
                    rank = source_spec.rank
                    axis = self._axis(params.get("axis", -1), rank)
                    if axis == rank - 1 and rank > 2:
                        permutation = self._channel_permutation(rank)
                        inverse = tuple(permutation.index(index) for index in range(rank))
                        lines.append(f"        {name}_input = {source}.permute{permutation}.contiguous()")
                        lines.append(f"        {name} = self.{name}({name}_input)")
                        lines.append(f"        {name} = {name}.permute{inverse}.contiguous()")
                    else:
                        lines.append(f"        {name} = self.{name}({source})")
                else:
                    lines.append(f"        {name} = self.{name}({source})")
                activation = params.get("activation", "linear")
                if definition == "layer.dense" and activation != "linear":
                    lines.append(f"        {name} = ({self._activation(str(activation))})({name})")
                variables[item.node_id] = name
            elif definition == "layer.embedding":
                lines.append(f"        {name} = self.{name}({source}.long())")
                variables[item.node_id] = name
            elif definition == "conv.conv2d":
                data_format = str(params.get("data_format", "channels_last"))
                kernel_size = self._spatial_tuple(params, "kernel_size", [3, 3])
                strides = self._spatial_tuple(params, "strides", [1, 1])
                dilation = self._spatial_tuple(params, "dilation_rate", [1, 1])
                padding = str(params.get("padding", "same"))
                conv_input = f"{name}_input"
                lines.append(f"        {conv_input} = {source}")
                if data_format == "channels_last":
                    lines.append(f"        {conv_input} = {conv_input}.permute(0, 3, 1, 2).contiguous()")
                if padding == "same":
                    lines.append(
                        f"        {conv_input} = _same_pad_2d({conv_input}, {kernel_size!r}, {strides!r}, {dilation!r})"
                    )
                lines.append(f"        {name} = self.{name}({conv_input})")
                if data_format == "channels_last":
                    lines.append(f"        {name} = {name}.permute(0, 2, 3, 1).contiguous()")
                activation = params.get("activation", "linear")
                if activation != "linear":
                    lines.append(f"        {name} = ({self._activation(str(activation))})({name})")
                variables[item.node_id] = name
            elif definition == "conv.max_pool2d":
                data_format = str(params.get("data_format", "channels_last"))
                pool_size = self._spatial_tuple(params, "pool_size", [2, 2])
                strides_value = params.get("strides")
                strides = (
                    pool_size
                    if strides_value is None
                    else self._spatial_tuple(params, "strides", [2, 2])
                )
                padding = str(params.get("padding", "valid"))
                pool_input = f"{name}_input"
                lines.append(f"        {pool_input} = {source}")
                if data_format == "channels_last":
                    lines.append(f"        {pool_input} = {pool_input}.permute(0, 3, 1, 2).contiguous()")
                if padding == "same":
                    lines.append(
                        f"        {pool_input} = _same_pad_2d({pool_input}, {pool_size!r}, {strides!r}, {(1, 1)!r}, float('-inf'))"
                    )
                lines.append(f"        {name} = self.{name}({pool_input})")
                if data_format == "channels_last":
                    lines.append(f"        {name} = {name}.permute(0, 2, 3, 1).contiguous()")
                variables[item.node_id] = name
            elif definition == "layer.activation":
                lines.append(f"        {name} = ({self._activation(str(params.get('activation', 'relu')))})({source})")
                variables[item.node_id] = name
            elif definition in {"recurrent.lstm", "recurrent.gru"}:
                lines.append(f"        {name}, _ = self.{name}({source})")
                if not params.get("return_sequences", False):
                    lines.append(f"        {name} = {name}[:, -1, :]")
                variables[item.node_id] = name
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(f"        {name} = {' + '.join(source_vars)}")
                variables[item.node_id] = name
            elif definition == "merge.concatenate":
                axis = params.get("axis", -1)
                lines.append(f"        {name} = torch.cat([{', '.join(source_vars)}], dim={axis!r})")
                variables[item.node_id] = name
            elif definition.startswith("output."):
                lines.append(f"        {name} = {source}")
                variables[item.node_id] = name
            else:
                raise ValueError(f"No PyTorch adapter registered for {definition}")

        outputs = [variables[node_id] for node_id in ir.output_node_ids]
        result = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
        lines.append(f"        return {result}")
        return "\n".join(lines) + "\n"
