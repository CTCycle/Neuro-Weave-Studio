# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from typing import Any, cast

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import (
    CompilationCatalog,
    CompilationOption,
    get_compilation_catalog,
)
from domain.graph import ModelGraph
from domain.metric_formula import MetricFormulaError, emit_metric_formula, parse_metric_formula
from domain.tensor import TensorSpec
from exporters.base import SourceExporter
from validation.output_resolution import output_selectors, resolved_outputs


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
    def generate(
        self,
        graph: ModelGraph,
        ir: CanonicalModelIR,
        catalog: NodeCatalog | None = None,
        *,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> str:
        catalog = catalog or get_node_catalog()
        compilation_catalog = compilation_catalog or get_compilation_catalog()
        nodes = {node.id: node for node in graph.nodes}
        summaries = self._summary_map(ir)
        native = graph.model_configuration.native.pytorch
        output_items = [
            output for output in resolved_outputs(graph, catalog) if output.node is not None
        ]

        def selected_value(configuration: dict[str, Any], output: Any, default: Any) -> Any:
            value = configuration.get("__default__", default)
            for selector in output_selectors(output):
                if selector and selector in configuration:
                    value = configuration[selector]
                    break
            return value

        def option(options: list[CompilationOption], identifier: str) -> CompilationOption:
            selected = next((item for item in options if item.id == identifier), None)
            if selected is None:
                raise ValueError(f"No compilation catalog option registered for {identifier}")
            implementation = selected.implementations.get("pytorch")
            if implementation is None or not implementation.supported or not implementation.adapter:
                raise ValueError(f"No PyTorch compilation adapter registered for {identifier}")
            return selected

        optimizer_option = option(compilation_catalog.optimizers, native.optimizer)
        optimizer_class = optimizer_option.implementations["pytorch"].adapter.rsplit(".", 1)[-1]  # type: ignore[union-attr]
        resolved_losses: dict[str, tuple[str, str]] = {}
        resolved_metrics: dict[str, list[tuple[str, str]]] = {}
        target_contracts: dict[str, str] = {}
        metric_target_contracts: dict[str, dict[str, str]] = {}
        for output in output_items:
            loss_id = selected_value(native.losses, output, None)
            if isinstance(loss_id, str):
                loss_option = option(compilation_catalog.losses, loss_id)
                loss_adapter = loss_option.implementations["pytorch"].adapter
                if loss_adapter is None:
                    raise ValueError(f"No PyTorch loss adapter registered for {loss_id}")
                resolved_losses[output.name] = (loss_id, loss_adapter)
                if loss_option.target_contract is not None:
                    target_contracts[output.name] = loss_option.target_contract
            metric_values = selected_value(native.metrics, output, [])
            if isinstance(metric_values, list):
                resolved_metrics[output.name] = []
                for metric_id in cast(list[Any], metric_values):
                    if not isinstance(metric_id, str):
                        continue
                    metric_option = option(compilation_catalog.metrics, metric_id)
                    metric_adapter = metric_option.implementations["pytorch"].adapter
                    if metric_adapter is None:
                        raise ValueError(f"No PyTorch metric adapter registered for {metric_id}")
                    resolved_metrics[output.name].append((metric_id, metric_adapter))
                    metric_target_contracts.setdefault(output.name, {})[metric_id] = (
                        metric_option.target_contract or "continuous_same_shape"
                    )
                    target_contracts.setdefault(
                        output.name, metric_option.target_contract or "continuous_same_shape"
                    )
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
            "def _same_pad_1d(value: Tensor, kernel_size: int, stride: int, dilation: int, fill_value: float = 0.0) -> Tensor:",
            "    length = value.shape[-1]",
            "    output_length = (length + stride - 1) // stride",
            "    effective_kernel = dilation * (kernel_size - 1) + 1",
            "    total = max((output_length - 1) * stride + effective_kernel - length, 0)",
            "    left = total // 2",
            "    return F.pad(value, (left, total - left), value=fill_value)",
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
                lines.append(
                    f"        self.{name} = nn.Dropout({float(params.get('rate', 0.5))!r})"
                )
            elif definition == "layer.flatten":
                lines.append(f"        self.{name} = nn.Flatten(start_dim=1)")
            elif definition == "layer.batch_norm":
                if source_spec is None:
                    raise ValueError(
                        f"Cannot infer the rank for batch normalization node {node.id}"
                    )
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
                    raise ValueError(
                        f"Cannot infer the shape for layer normalization node {node.id}"
                    )
                axes = self._normalization_axes(params.get("axis", [-1]), source_spec.rank)
                shape = cast(list[Any], source_spec.shape)
                normalized_shape = [shape[index] for index in axes]
                if not all(isinstance(item, int) for item in normalized_shape):
                    raise ValueError(
                        f"LayerNorm node {node.id} requires known normalized dimensions"
                    )
                if not bool(params.get("center", True)) or not bool(params.get("scale", True)):
                    raise ValueError("PyTorch LayerNorm requires center and scale")
                lines.append(
                    f"        self.{name} = nn.LayerNorm({normalized_shape!r}, eps={float(params.get('epsilon', 0.001))!r})"
                )
            elif definition == "layer.embedding":
                lines.append(
                    f"        self.{name} = nn.Embedding({int(params.get('vocabulary_size', 10000))!r}, {int(params.get('embedding_dim', 128))!r})"
                )
            elif definition == "conv.conv1d":
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"PyTorch Conv1D node {node.id} requires a rank-3 input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = 2 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                kernel_size = params.get("kernel_size", [3])
                strides = params.get("strides", [1])
                dilation = params.get("dilation_rate", [1])
                if not isinstance(kernel_size, list) or len(kernel_size) != 1:
                    raise ValueError("PyTorch Conv1D expects one kernel_size value")
                if not isinstance(strides, list) or len(strides) != 1:
                    raise ValueError("PyTorch Conv1D expects one strides value")
                if not isinstance(dilation, list) or len(dilation) != 1:
                    raise ValueError("PyTorch Conv1D expects one dilation_rate value")
                kernel = int(kernel_size[0])
                stride = int(strides[0])
                dilation_value = int(dilation[0])
                filters = int(params.get("filters", 32))
                groups = int(params.get("groups", 1))
                bias = bool(params.get("use_bias", True))
                if input_channels is None and groups != 1:
                    raise ValueError("PyTorch LazyConv1d cannot infer grouped input channels")
                module = "nn.Conv1d" if input_channels is not None else "nn.LazyConv1d"
                constructor = (
                    f"{input_channels!r}, {filters!r}"
                    if input_channels is not None
                    else f"{filters!r}"
                )
                lines.append(
                    f"        self.{name} = {module}({constructor}, kernel_size={kernel!r}, stride={stride!r}, padding=0, dilation={dilation_value!r}, groups={groups!r}, bias={bias!r})"
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
                if (
                    params.get("activation", "tanh") != "tanh"
                    or params.get("recurrent_activation", "sigmoid") != "sigmoid"
                ):
                    raise ValueError(
                        "PyTorch recurrent export supports the canonical tanh/sigmoid activations only"
                    )
                if (
                    float(params.get("dropout", 0.0)) != 0.0
                    or float(params.get("recurrent_dropout", 0.0)) != 0.0
                ):
                    raise ValueError("PyTorch recurrent export does not support recurrent dropout")
                if (
                    params.get("go_backwards", False)
                    or params.get("stateful", False)
                    or params.get("unroll", False)
                ):
                    raise ValueError(
                        "PyTorch recurrent export does not support go_backwards, stateful, or unroll"
                    )
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
                "    def forward(self, inputs: Tensor | dict[str, Tensor], return_intermediates: bool = False) -> Tensor | tuple[Tensor, ...] | tuple[Tensor | tuple[Tensor, ...], dict[str, dict[str, Tensor]]]:",
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

            manifest = catalog.get(definition, node.definition_version)
            if manifest.operation is not None:
                lines.append(
                    f"        {name} = {emit_operation_source(manifest, 'pytorch', source_vars, params)}"
                )
                variables[item.node_id] = name
            elif manifest.semantic_role == "input":
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
                        lines.append(
                            f"        {name}_input = {source}.permute{permutation}.contiguous()"
                        )
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
            elif definition == "conv.conv1d":
                data_format = str(params.get("data_format", "channels_last"))
                kernel_size = params.get("kernel_size", [3])
                strides = params.get("strides", [1])
                dilation = params.get("dilation_rate", [1])
                kernel = int(kernel_size[0])
                stride = int(strides[0])
                dilation_value = int(dilation[0])
                padding = str(params.get("padding", "same"))
                conv_input = f"{name}_input"
                lines.append(f"        {conv_input} = {source}")
                if data_format == "channels_last":
                    lines.append(
                        f"        {conv_input} = {conv_input}.permute(0, 2, 1).contiguous()"
                    )
                if padding == "same":
                    lines.append(
                        f"        {conv_input} = _same_pad_1d({conv_input}, {kernel!r}, {stride!r}, {dilation_value!r})"
                    )
                lines.append(f"        {name} = self.{name}({conv_input})")
                if data_format == "channels_last":
                    lines.append(f"        {name} = {name}.permute(0, 2, 1).contiguous()")
                activation = params.get("activation", "linear")
                if activation != "linear":
                    lines.append(f"        {name} = ({self._activation(str(activation))})({name})")
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
                    lines.append(
                        f"        {conv_input} = {conv_input}.permute(0, 3, 1, 2).contiguous()"
                    )
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
                    lines.append(
                        f"        {pool_input} = {pool_input}.permute(0, 3, 1, 2).contiguous()"
                    )
                if padding == "same":
                    lines.append(
                        f"        {pool_input} = _same_pad_2d({pool_input}, {pool_size!r}, {strides!r}, {(1, 1)!r}, float('-inf'))"
                    )
                lines.append(f"        {name} = self.{name}({pool_input})")
                if data_format == "channels_last":
                    lines.append(f"        {name} = {name}.permute(0, 2, 3, 1).contiguous()")
                variables[item.node_id] = name
            elif definition == "layer.activation":
                lines.append(
                    f"        {name} = ({self._activation(str(params.get('activation', 'relu')))})({source})"
                )
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
                lines.append(
                    f"        {name} = torch.cat([{', '.join(source_vars)}], dim={axis!r})"
                )
                variables[item.node_id] = name
            elif manifest.semantic_role == "output_adapter":
                lines.append(f"        {name} = {source}")
                variables[item.node_id] = name
            else:
                raise ValueError(f"No PyTorch adapter registered for {definition}")

        outputs = [variables[binding.node_id] for binding in ir.output_bindings]
        result = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
        trace = "{" + ", ".join(
            f"{item.node_id!r}: {{'output': {variables[item.node_id]}}}"
            for item in ir.ordered_nodes
        ) + "}"
        lines.append(f"        if return_intermediates: return {result}, {trace}")
        lines.append(f"        return {result}")
        optimizer_parameters = {
            parameter.id: native.optimizer_parameters.get(parameter.id, parameter.default)
            for parameter in optimizer_option.parameters
        }
        optimizer_kwargs = [
            f"        lr={optimizer_parameters['learning_rate']!r},",
        ]
        if native.optimizer == "adam":
            optimizer_kwargs.extend(
                [
                    f"        betas=({optimizer_parameters['beta_1']!r}, {optimizer_parameters['beta_2']!r}),",
                    f"        eps={optimizer_parameters['epsilon']!r},",
                ]
            )
        elif native.optimizer == "adamw":
            optimizer_kwargs.extend(
                [
                    f"        betas=({optimizer_parameters['beta_1']!r}, {optimizer_parameters['beta_2']!r}),",
                    f"        eps={optimizer_parameters['epsilon']!r},",
                    f"        weight_decay={optimizer_parameters['weight_decay']!r},",
                ]
            )
        elif native.optimizer == "sgd":
            optimizer_kwargs.extend(
                [
                    f"        momentum={optimizer_parameters['momentum']!r},",
                    f"        nesterov={optimizer_parameters['nesterov']!r},",
                ]
            )
        elif native.optimizer == "rmsprop":
            optimizer_kwargs.extend(
                [
                    f"        alpha={optimizer_parameters['rho']!r},",
                    f"        momentum={optimizer_parameters['momentum']!r},",
                    f"        eps={optimizer_parameters['epsilon']!r},",
                    f"        centered={optimizer_parameters['centered']!r},",
                ]
            )

        custom_loss_lines: list[str] = []
        custom_metric_lines: list[str] = []
        custom_loss_entries: list[str] = []
        custom_metric_entries: list[str] = []
        referenced_options = {
            option.id: option
            for option in (*compilation_catalog.losses, *compilation_catalog.metrics)
            if option.origin == "custom"
        }
        for identifier, _adapter in [*resolved_losses.values(), *[
            item for values in resolved_metrics.values() for item in values
        ]]:
            custom_option = referenced_options.get(identifier)
            if custom_option is None:
                continue
            function_name = "_custom_" + "".join(
                char if char.isalnum() else "_" for char in identifier
            )
            try:
                expression = emit_metric_formula(
                    parse_metric_formula(custom_option.formula or ""),
                    "pytorch",
                    target="target",
                    prediction="prediction",
                )
            except MetricFormulaError as exc:
                raise ValueError(f"Invalid custom formula for {identifier}: {exc}") from exc
            function_lines = [
                "",
                "",
                f"def {function_name}(prediction: Tensor, target: Tensor) -> Tensor:",
                f"    return {expression}",
            ]
            if identifier in {item[0] for item in resolved_losses.values()}:
                custom_loss_lines.extend(function_lines)
                custom_loss_entries.append(f"    {identifier!r}: {function_name},")
            else:
                custom_metric_lines.extend(function_lines)
                custom_metric_entries.append(f"    {identifier!r}: {function_name},")

        lines.extend(
            [
                "",
                "",
                "def _safe_probabilities(value: Tensor) -> Tensor:",
                "    return value.to(torch.float32).clamp(min=1e-7, max=1.0 - 1e-7)",
                "",
                "",
                "def _loss_mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.mse_loss(prediction, target.to(prediction.dtype))",
                "",
                "",
                "def _loss_mean_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.l1_loss(prediction, target.to(prediction.dtype))",
                "",
                "",
                "def _loss_binary_crossentropy(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.binary_cross_entropy(_safe_probabilities(prediction), target.to(torch.float32))",
                "",
                "",
                "def _loss_categorical_crossentropy(prediction: Tensor, target: Tensor) -> Tensor:",
                "    probabilities = _safe_probabilities(prediction)",
                "    return -(target.to(torch.float32) * probabilities.log()).sum(dim=-1).mean()",
                "",
                "",
                "def _loss_sparse_categorical_crossentropy(prediction: Tensor, target: Tensor) -> Tensor:",
                "    probabilities = _safe_probabilities(prediction)",
                "    log_probabilities = probabilities.log().reshape(-1, probabilities.shape[-1])",
                "    return F.nll_loss(log_probabilities, target.to(torch.long).reshape(-1))",
                "",
                "",
                "def _loss_huber(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.huber_loss(prediction, target.to(prediction.dtype))",
                "",
                "",
                "def _loss_kullback_leibler_divergence(prediction: Tensor, target: Tensor) -> Tensor:",
                "    probabilities = _safe_probabilities(prediction)",
                "    distribution = target.to(torch.float32).clamp_min(1e-7)",
                "    return F.kl_div(probabilities.log(), distribution, reduction='batchmean')",
                "",
                "",
                "def _classification_labels(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:",
                "    if prediction.shape[-1] == 1:",
                "        return (prediction >= 0.5).to(torch.long).reshape(-1), (target >= 0.5).to(torch.long).reshape(-1)",
                "    predicted = prediction.argmax(dim=-1).reshape(-1)",
                "    expected = target.argmax(dim=-1).reshape(-1) if target.shape == prediction.shape else target.to(torch.long).reshape(-1)",
                "    return predicted, expected",
                "",
                "",
                "def _metric_accuracy(prediction: Tensor, target: Tensor) -> Tensor:",
                "    predicted, expected = _classification_labels(prediction, target)",
                "    return (predicted == expected).to(torch.float32).mean()",
                "",
                "",
                "def _metric_binary_accuracy(prediction: Tensor, target: Tensor) -> Tensor:",
                "    predicted = (prediction >= 0.5).to(torch.long)",
                "    expected = (target >= 0.5).to(torch.long)",
                "    return (predicted == expected).to(torch.float32).mean()",
                "",
                "",
                "def _metric_categorical_accuracy(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return (prediction.argmax(dim=-1) == target.argmax(dim=-1)).to(torch.float32).mean()",
                "",
                "",
                "def _metric_sparse_categorical_accuracy(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return (prediction.argmax(dim=-1) == target.to(torch.long)).to(torch.float32).mean()",
                "",
                "",
                "def _metric_precision(prediction: Tensor, target: Tensor) -> Tensor:",
                "    predicted, expected = _classification_labels(prediction, target)",
                "    true_positive = ((predicted == 1) & (expected == 1)).to(torch.float32).sum()",
                "    predicted_positive = (predicted == 1).to(torch.float32).sum()",
                "    return true_positive / predicted_positive.clamp_min(1.0)",
                "",
                "",
                "def _metric_recall(prediction: Tensor, target: Tensor) -> Tensor:",
                "    predicted, expected = _classification_labels(prediction, target)",
                "    true_positive = ((predicted == 1) & (expected == 1)).to(torch.float32).sum()",
                "    actual_positive = (expected == 1).to(torch.float32).sum()",
                "    return true_positive / actual_positive.clamp_min(1.0)",
                "",
                "",
                "def _metric_auc(prediction: Tensor, target: Tensor) -> Tensor:",
                "    scores = prediction[..., 0] if prediction.shape[-1] == 1 else prediction[..., 1]",
                "    labels = target[..., 0] if target.shape[-1] == 1 else target.argmax(dim=-1).to(prediction.dtype)",
                "    positive = scores[labels.reshape(-1) > 0.5]",
                "    negative = scores[labels.reshape(-1) <= 0.5]",
                "    if positive.numel() == 0 or negative.numel() == 0:",
                "        return scores.new_tensor(0.5)",
                "    comparisons = positive.reshape(-1, 1) - negative.reshape(1, -1)",
                "    return (comparisons > 0).to(torch.float32).mean() + 0.5 * (comparisons == 0).to(torch.float32).mean()",
                "",
                "",
                "def _metric_mean_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.l1_loss(prediction, target.to(prediction.dtype))",
                "",
                "",
                "def _metric_mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.mse_loss(prediction, target.to(prediction.dtype))",
                "",
                "",
                "def _metric_root_mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.mse_loss(prediction, target.to(prediction.dtype)).sqrt()",
                *custom_loss_lines,
                *custom_metric_lines,
                "",
                "",
                "_LOSS_ADAPTERS = {",
                "    'mean_squared_error': _loss_mean_squared_error,",
                "    'mean_absolute_error': _loss_mean_absolute_error,",
                "    'binary_crossentropy': _loss_binary_crossentropy,",
                "    'categorical_crossentropy': _loss_categorical_crossentropy,",
                "    'sparse_categorical_crossentropy': _loss_sparse_categorical_crossentropy,",
                "    'huber': _loss_huber,",
                "    'kullback_leibler_divergence': _loss_kullback_leibler_divergence,",
                *custom_loss_entries,
                "}",
                "",
                "",
                "_METRIC_ADAPTERS = {",
                "    'accuracy': _metric_accuracy,",
                "    'binary_accuracy': _metric_binary_accuracy,",
                "    'categorical_accuracy': _metric_categorical_accuracy,",
                "    'sparse_categorical_accuracy': _metric_sparse_categorical_accuracy,",
                "    'precision': _metric_precision,",
                "    'recall': _metric_recall,",
                "    'auc': _metric_auc,",
                "    'mean_absolute_error': _metric_mean_absolute_error,",
                "    'mean_squared_error': _metric_mean_squared_error,",
                "    'root_mean_squared_error': _metric_root_mean_squared_error,",
                *custom_metric_entries,
                "}",
                "",
                "",
                "def build_raw_model() -> GeneratedModel:",
                "    return GeneratedModel()",
                "",
                "",
                "def compile_model(model: GeneratedModel) -> GeneratedModel:",
                f"    if not {native.compile_enabled!r}:",
                "        return model",
                "    return torch.compile(",
                "        model,",
                f"        mode={native.mode!r},",
                f"        fullgraph={native.fullgraph!r},",
                f"        dynamic={native.dynamic!r},",
                "    )",
                "",
                "",
                "def build_optimizer(model: GeneratedModel) -> torch.optim.Optimizer | None:",
                "    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]",
                "    if not parameters:",
                "        return None",
                f"    return torch.optim.{optimizer_class}(",
                "        parameters,",
                *optimizer_kwargs,
                "    )",
                "",
                "",
                "def configured_loss_ids() -> dict[str, str]:",
                f"    return { {output_id: loss_id for output_id, (loss_id, _) in resolved_losses.items()}!r}",
                "",
                "",
                "def configured_losses() -> dict[str, object]:",
                f"    return {{output_id: _LOSS_ADAPTERS[loss_id] for output_id, loss_id in { {output_id: loss_id for output_id, (loss_id, _) in resolved_losses.items()}!r}.items()}}",
                "",
                "",
                "def configured_metric_ids() -> dict[str, list[str]]:",
                f"    return { {output_id: [metric_id for metric_id, _ in values] for output_id, values in resolved_metrics.items()}!r}",
                "",
                "",
                "def configured_metrics() -> dict[str, dict[str, object]]:",
                f"    return {{output_id: {{metric_id: _METRIC_ADAPTERS[metric_id] for metric_id in metric_ids}} for output_id, metric_ids in { {output_id: [metric_id for metric_id, _ in values] for output_id, values in resolved_metrics.items()}!r}.items()}}",
                "",
                "",
                "def configured_target_contracts() -> dict[str, str]:",
                f"    return {target_contracts!r}",
                "",
                "",
                "def configured_metric_target_contracts() -> dict[str, dict[str, str]]:",
                f"    return {metric_target_contracts!r}",
                "",
                "",
                "def configured_compilation() -> dict[str, object]:",
                "    return {",
                f"        'optimizer_id': {native.optimizer!r},",
                f"        'loss_ids': { {output_id: loss_id for output_id, (loss_id, _) in resolved_losses.items()}!r},",
                f"        'metric_ids': { {output_id: [metric_id for metric_id, _ in values] for output_id, values in resolved_metrics.items()}!r},",
                f"        'target_contracts': {target_contracts!r},",
                f"        'metric_target_contracts': {metric_target_contracts!r},",
                "        'training': 'not_run',",
                "    }",
                "",
                "",
                "def build_compilation(model: GeneratedModel) -> dict[str, object]:",
                "    configuration = configured_compilation()",
                "    configuration.update({",
                "        'optimizer': build_optimizer(model),",
                "        'losses': configured_losses(),",
                "        'metrics': configured_metrics(),",
                "    })",
                "    return configuration",
                "",
                "",
                "def build_model() -> GeneratedModel:",
                "    return compile_model(build_raw_model())",
                "",
            ]
        )
        return "\n".join(lines) + "\n"
