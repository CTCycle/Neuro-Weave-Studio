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
from domain.compilation_plan import CompilationObject
from domain.graph import ModelGraph
from domain.metric_formula import MetricFormulaError, emit_metric_formula, parse_metric_formula
from domain.ports import materialized_outputs
from domain.tensor import TensorSpec
from exporters.base import SourceExporter
from validation.output_resolution import output_selectors, resolved_outputs
from validation.safe_expression import normalize_index_expression, validate_safe_expression


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
            raise ValueError(f"PyTorch {key} expects {dimensions} values")
        values = cast(list[Any], value)
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in values):
            raise ValueError(f"PyTorch {key} expects positive integer values")
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
            raise ValueError(f"PyTorch {key} expects {dimensions} values")
        values = cast(list[Any], value)
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
            raise ValueError(f"PyTorch {key} expects non-negative integer values")
        return tuple(int(item) for item in values)

    # -------------------------------------------------------------------------
    @staticmethod
    def _spatial_pairs(parameters: dict[str, Any], key: str, dimensions: int) -> tuple[tuple[int, int], ...]:
        value = parameters.get(key)
        if not isinstance(value, list) or len(value) != dimensions:
            raise ValueError(f"PyTorch {key} expects {dimensions} pairs")
        pairs: list[tuple[int, int]] = []
        for item in cast(list[Any], value):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(not isinstance(entry, int) or isinstance(entry, bool) or entry < 0 for entry in item)
            ):
                raise ValueError(f"PyTorch {key} expects non-negative pairs")
            pairs.append((int(item[0]), int(item[1])))
        return tuple(pairs)

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
    def _index_expression(value: object) -> str:
        expression = normalize_index_expression(value if isinstance(value, str) else ":")
        validate_safe_expression(expression)
        return expression

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
        compilation = native.compilation
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

        optimizer_value = compilation.optimizer
        optimizer_option = option(compilation_catalog.optimizers, optimizer_value.id)
        optimizer_class = optimizer_option.implementations["pytorch"].adapter.rsplit(".", 1)[-1]  # type: ignore[union-attr]
        resolved_losses: dict[str, tuple[str, str]] = {}
        resolved_metrics: dict[str, list[tuple[str, str]]] = {}
        loss_configurations: dict[str, CompilationObject] = {}
        metric_configurations: dict[str, list[CompilationObject]] = {}
        target_contracts: dict[str, str] = {}
        metric_target_contracts: dict[str, dict[str, str]] = {}
        for output in output_items:
            loss_value = selected_value(compilation.losses, output, None)
            if isinstance(loss_value, CompilationObject):
                loss_id = loss_value.id
                loss_option = option(compilation_catalog.losses, loss_id)
                loss_adapter = loss_option.implementations["pytorch"].adapter
                if loss_adapter is None:
                    raise ValueError(f"No PyTorch loss adapter registered for {loss_id}")
                resolved_losses[output.name] = (loss_id, loss_adapter)
                loss_configurations[output.name] = loss_value
                if loss_option.target_contract is not None:
                    target_contracts[output.name] = loss_option.target_contract
            metric_values = selected_value(compilation.metrics, output, [])
            if isinstance(metric_values, list):
                resolved_metrics[output.name] = []
                for metric_value in cast(list[Any], metric_values):
                    if not isinstance(metric_value, CompilationObject):
                        continue
                    metric_id = metric_value.id
                    metric_option = option(compilation_catalog.metrics, metric_id)
                    metric_adapter = metric_option.implementations["pytorch"].adapter
                    if metric_adapter is None:
                        raise ValueError(f"No PyTorch metric adapter registered for {metric_id}")
                    resolved_metrics[output.name].append((metric_id, metric_adapter))
                    metric_configurations.setdefault(output.name, []).append(metric_value)
                    metric_target_contracts.setdefault(output.name, {})[metric_id] = (
                        metric_option.target_contract or "continuous_same_shape"
                    )
                    target_contracts.setdefault(
                        output.name, metric_option.target_contract or "continuous_same_shape"
                    )
        loss_payload = {
            output_id: {"id": value.id, "parameters": dict(value.parameters)}
            for output_id, value in loss_configurations.items()
        }
        metric_payload = {
            output_id: [
                {"id": value.id, "parameters": dict(value.parameters)}
                for value in values
            ]
            for output_id, values in metric_configurations.items()
        }
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
            "def _same_pad_nd(value: Tensor, kernel_size: tuple[int, ...], stride: tuple[int, ...], fill_value: float = 0.0, dilation: tuple[int, ...] | None = None) -> Tensor:",
            "    pairs = []",
            "    dimensions = len(kernel_size)",
            "    dilation = dilation or (1,) * dimensions",
            "    for index in range(dimensions):",
            "        size = value.shape[-dimensions + index]",
            "        output_size = (size + stride[index] - 1) // stride[index]",
            "        effective_kernel = dilation[index] * (kernel_size[index] - 1) + 1",
            "        total = max((output_size - 1) * stride[index] + effective_kernel - size, 0)",
            "        left = total // 2",
            "        pairs.append((left, total - left))",
            "    padding = [item for pair in reversed(pairs) for item in pair]",
            "    return F.pad(value, padding, value=fill_value)",
            "",
            "",
            "def _crop_transposed_same(value: Tensor, input_shape: torch.Size, stride: tuple[int, ...], output_padding: tuple[int, ...]) -> Tensor:",
            "    slices = [slice(None)] * (len(stride) + 2)",
            "    for index, multiplier in enumerate(stride):",
            "        slices[index + 2] = slice(0, input_shape[index + 2] * multiplier + output_padding[index])",
            "    return value[tuple(slices)]",
            "",
            "",
            "def _pool_nd(value: Tensor, pool_size: tuple[int, ...], strides: tuple[int, ...], padding: str, data_format: str, mode: str) -> Tensor:",
            "    dimensions = len(pool_size)",
            "    channels_last = data_format == 'channels_last'",
            "    if channels_last:",
            "        value = value.permute((0, dimensions + 1, *range(1, dimensions + 1))).contiguous()",
            "    if padding == 'same':",
            "        value = _same_pad_nd(value, pool_size, strides, float('-inf') if mode == 'max' else 0.0)",
            "    operation = getattr(F, f'{mode}_pool{dimensions}d')",
            "    current = operation(value, kernel_size=pool_size, stride=strides, padding=0)",
            "    if channels_last:",
            "        current = current.permute((0, *range(2, dimensions + 2), 1)).contiguous()",
            "    return current",
            "",
            "",
            "def _global_pool_nd(value: Tensor, dimensions: int, data_format: str, mode: str, keepdims: bool) -> Tensor:",
            "    channels_last = data_format == 'channels_last'",
            "    if channels_last:",
            "        value = value.permute((0, dimensions + 1, *range(1, dimensions + 1))).contiguous()",
            "    axes = tuple(range(2, dimensions + 2))",
            "    current = value.mean(dim=axes, keepdim=keepdims) if mode == 'average' else torch.amax(value, dim=axes, keepdim=keepdims)",
            "    if channels_last and keepdims:",
            "        current = current.permute((0, *range(2, dimensions + 2), 1)).contiguous()",
            "    return current.reshape(current.shape[0], current.shape[1]) if not keepdims else current",
            "",
            "",
            "def _upsample_nd(value: Tensor, size: tuple[int, ...], data_format: str, interpolation: str) -> Tensor:",
            "    dimensions = len(size)",
            "    channels_last = data_format == 'channels_last'",
            "    if channels_last:",
            "        value = value.permute((0, dimensions + 1, *range(1, dimensions + 1))).contiguous()",
            "    target = tuple(value.shape[-dimensions + index] * size[index] for index in range(dimensions))",
            "    if interpolation == 'bilinear' and dimensions == 2:",
            "        current = F.interpolate(value, size=target, mode='bilinear', align_corners=False)",
            "    else:",
            "        current = F.interpolate(value, size=target, mode='nearest')",
            "    if channels_last:",
            "        current = current.permute((0, *range(2, dimensions + 2), 1)).contiguous()",
            "    return current",
            "",
            "",
            "def _pad_nd(value: Tensor, pairs: tuple[tuple[int, int], ...], data_format: str) -> Tensor:",
            "    dimensions = len(pairs)",
            "    channels_last = data_format == 'channels_last'",
            "    if channels_last:",
            "        value = value.permute((0, dimensions + 1, *range(1, dimensions + 1))).contiguous()",
            "    padding = [item for pair in reversed(pairs) for item in pair]",
            "    current = F.pad(value, padding)",
            "    if channels_last:",
            "        current = current.permute((0, *range(2, dimensions + 2), 1)).contiguous()",
            "    return current",
            "",
            "",
            "def _set_lstm_forget_bias(layer: nn.LSTM) -> None:",
            "    with torch.no_grad():",
            "        hidden_size = layer.hidden_size",
            "        layer.bias_ih_l0[hidden_size:2 * hidden_size].fill_(1.0)",
            "",
            "",
            "def _initialize(shape, initializer, dtype):",
            "    identifier = initializer.get('id', 'zeros') if isinstance(initializer, dict) else initializer",
            "    if identifier in (None, 'zeros'): return torch.zeros(shape, dtype=dtype)",
            "    if identifier == 'ones': return torch.ones(shape, dtype=dtype)",
            "    if identifier in ('random_normal', 'normal'): return torch.randn(shape, dtype=dtype)",
            "    if identifier in ('random_uniform', 'uniform'): return torch.rand(shape, dtype=dtype)",
            "    return torch.zeros(shape, dtype=dtype)",
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

            if definition == "input.constant":
                shape = params.get("shape", [1])
                dtype = params.get("dtype", "float32")
                value = params.get("value", 0)
                lines.append(
                    f"        self.register_buffer({name!r}, torch.as_tensor({value!r}, dtype=getattr(torch, {dtype!r})).expand({tuple(shape)!r}))"
                )
            elif definition == "input.parameter":
                shape = params.get("shape", [1])
                dtype = params.get("dtype", "float32")
                initializer = params.get("initializer", "zeros")
                expression = f"_initialize({tuple(shape)!r}, {initializer!r}, getattr(torch, {dtype!r}))"
                if bool(params.get("trainable", True)):
                    lines.append(
                        f"        self.{name} = nn.Parameter({expression}, requires_grad=True)"
                    )
                else:
                    lines.append(f"        self.register_buffer({name!r}, {expression})")
            elif definition == "layer.prelu":
                if source_spec is None:
                    raise ValueError(f"PyTorch PReLU node {node.id} requires a known input shape")
                alpha_shape = list(source_spec.shape[1:])
                if not all(isinstance(value, int) for value in alpha_shape):
                    raise ValueError(f"PyTorch PReLU node {node.id} requires known input dimensions")
                shared_axes = params.get("shared_axes")
                if shared_axes is not None:
                    if not isinstance(shared_axes, list):
                        raise ValueError("PyTorch PReLU shared_axes must be a list or null")
                    for shared_axis in shared_axes:
                        if not isinstance(shared_axis, int) or not 1 <= shared_axis < source_spec.rank:
                            raise ValueError("PyTorch PReLU shared_axes must contain non-batch axes")
                        alpha_shape[shared_axis - 1] = 1
                lines.append(
                    f"        self.{name}_alpha = nn.Parameter(_initialize({alpha_shape!r}, {params.get('alpha_initializer', 'zeros')!r}, torch.float32))"
                )
                lines.append(f"        self.{name}_alpha_broadcast = {(1, *alpha_shape)!r}")
            elif definition == "layer.rms_norm":
                if source_spec is None:
                    raise ValueError(f"PyTorch RMSNormalization node {node.id} requires a known input shape")
                axis = self._axis(params.get("axis", -1), source_spec.rank)
                dimension = self._static_dimension(source_spec, axis)
                if dimension is None:
                    raise ValueError(f"PyTorch RMSNormalization node {node.id} requires a known normalized dimension")
                broadcast = [1] * source_spec.rank
                broadcast[axis] = -1
                lines.append(
                    f"        self.{name}_scale = nn.Parameter(torch.ones([{dimension}], dtype=torch.float32))"
                )
                lines.append(f"        self.{name}_axis = {axis}")
                lines.append(f"        self.{name}_broadcast = {tuple(broadcast)!r}")
                lines.append(f"        self.{name}_epsilon = {float(params.get('epsilon', 1e-6))!r}")
            elif definition == "layer.unit_norm":
                if source_spec is None:
                    raise ValueError(f"PyTorch UnitNormalization node {node.id} requires a known input rank")
                lines.append(f"        self.{name}_axis = {self._axis(params.get('axis', -1), source_spec.rank)}")
            elif definition == "layer.group_norm":
                if source_spec is None:
                    raise ValueError(f"PyTorch GroupNormalization node {node.id} requires a known input shape")
                rank = source_spec.rank
                axis = self._axis(params.get("axis", -1), rank)
                channels = self._static_dimension(source_spec, axis)
                groups = int(params.get("groups", 32))
                if channels is None:
                    raise ValueError(f"PyTorch GroupNormalization node {node.id} requires a known channel dimension")
                if channels % groups != 0:
                    raise ValueError("PyTorch GroupNormalization channels must be divisible by groups")
                lines.append(f"        self.{name}_axis = {axis}")
                lines.append(f"        self.{name}_groups = {groups}")
                lines.append(f"        self.{name}_epsilon = {float(params.get('epsilon', 0.001))!r}")
                lines.append(
                    f"        self.{name}_gamma = nn.Parameter(_initialize([{channels}], {params.get('gamma_initializer', 'ones')!r}, torch.float32))"
                    if bool(params.get("scale", True))
                    else f"        self.{name}_gamma = None"
                )
                lines.append(
                    f"        self.{name}_beta = nn.Parameter(_initialize([{channels}], {params.get('beta_initializer', 'zeros')!r}, torch.float32))"
                    if bool(params.get("center", True))
                    else f"        self.{name}_beta = None"
                )
            elif definition == "layer.dense":
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
            elif definition == "conv.conv3d":
                if source_spec is None or source_spec.rank != 5:
                    raise ValueError(f"PyTorch Conv3D node {node.id} requires a rank-5 input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = 4 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                kernel_size = self._spatial_values(params, "kernel_size", 3, [3, 3, 3])
                strides = self._spatial_values(params, "strides", 3, [1, 1, 1])
                dilation = self._spatial_values(params, "dilation_rate", 3, [1, 1, 1])
                filters = int(params.get("filters", 32))
                groups = int(params.get("groups", 1))
                bias = bool(params.get("use_bias", True))
                if input_channels is None and groups != 1:
                    raise ValueError("PyTorch LazyConv3d cannot infer grouped input channels")
                module = "nn.Conv3d" if input_channels is not None else "nn.LazyConv3d"
                constructor = (
                    f"{input_channels!r}, {filters!r}"
                    if input_channels is not None
                    else f"{filters!r}"
                )
                lines.append(
                    f"        self.{name} = {module}({constructor}, kernel_size={kernel_size!r}, stride={strides!r}, padding=0, dilation={dilation!r}, groups={groups!r}, bias={bias!r})"
                )
            elif definition == "conv.depthwise":
                if source_spec is None:
                    raise ValueError(f"PyTorch depthwise node {node.id} requires a known input shape")
                dimensions = int(params.get("dimensions", 2))
                if dimensions not in {1, 2} or source_spec.rank != dimensions + 2:
                    raise ValueError(f"PyTorch depthwise node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = dimensions + 1 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"PyTorch depthwise node {node.id} requires known input channels")
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                depth_multiplier = int(params.get("depth_multiplier", 1))
                module = f"nn.Conv{dimensions}d"
                output_channels = input_channels * depth_multiplier
                lines.append(
                    f"        self.{name} = {module}({input_channels!r}, {output_channels!r}, kernel_size={kernel_size!r}, stride={strides!r}, padding=0, dilation={dilation!r}, groups={input_channels!r}, bias={bool(params.get('use_bias', True))!r})"
                )
            elif definition == "conv.separable":
                if source_spec is None:
                    raise ValueError(f"PyTorch separable node {node.id} requires a known input shape")
                dimensions = int(params.get("dimensions", 2))
                if dimensions not in {1, 2} or source_spec.rank != dimensions + 2:
                    raise ValueError(f"PyTorch separable node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = dimensions + 1 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"PyTorch separable node {node.id} requires known input channels")
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                depth_multiplier = int(params.get("depth_multiplier", 1))
                filters = int(params.get("filters", 32))
                depth_channels = input_channels * depth_multiplier
                module = f"nn.Conv{dimensions}d"
                lines.append(
                    f"        self.{name}_depthwise = {module}({input_channels!r}, {depth_channels!r}, kernel_size={kernel_size!r}, stride={strides!r}, padding=0, dilation={dilation!r}, groups={input_channels!r}, bias=False)"
                )
                lines.append(
                    f"        self.{name}_pointwise = {module}({depth_channels!r}, {filters!r}, kernel_size={((1,) * dimensions)!r}, stride={((1,) * dimensions)!r}, padding=0, bias={bool(params.get('use_bias', True))!r})"
                )
            elif definition == "conv.transposed":
                if source_spec is None:
                    raise ValueError(f"PyTorch transposed convolution node {node.id} requires a known input shape")
                dimensions = int(params.get("dimensions", 2))
                if dimensions not in {1, 2, 3} or source_spec.rank != dimensions + 2:
                    raise ValueError(f"PyTorch transposed convolution node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = dimensions + 1 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"PyTorch transposed convolution node {node.id} requires known input channels")
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                output_padding_value = params.get("output_padding")
                output_padding = (
                    (0,) * dimensions
                    if output_padding_value is None
                    else self._spatial_nonnegative(params, "output_padding", dimensions, [0] * dimensions)
                )
                filters = int(params.get("filters", 32))
                lines.append(
                    f"        self.{name} = nn.ConvTranspose{dimensions}d({input_channels!r}, {filters!r}, kernel_size={kernel_size!r}, stride={strides!r}, padding=0, output_padding={output_padding!r}, groups=1, dilation={dilation!r}, bias={bool(params.get('use_bias', True))!r})"
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
            elif definition in {
                "recurrent.simple_rnn",
                "recurrent.bidirectional",
                "recurrent.stack",
            }:
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
                        raise ValueError("PyTorch bidirectional export does not expose recurrent state ports")
                if definition == "recurrent.stack":
                    if str(params.get("merge_mode", "concat")) != "concat":
                        raise ValueError("Recurrent stack merge_mode must remain concat")
                    if params.get("return_state", False):
                        raise ValueError("PyTorch recurrent stack does not expose recurrent state ports")
                recurrent_class = {"simple_rnn": "RNN", "lstm": "LSTM", "gru": "GRU"}[cell]
                constructor = (
                    f"{input_size!r}, {units!r}, batch_first=True, bias={bool(params.get('use_bias', True))!r}"
                )
                if cell == "simple_rnn":
                    constructor += ", nonlinearity='tanh'"
                if definition == "recurrent.bidirectional":
                    constructor += ", bidirectional=True"
                if definition == "recurrent.stack":
                    layers = int(params.get("layers", 2))
                    if layers < 1:
                        raise ValueError("Recurrent stack layers must be positive")
                    constructor += f", num_layers={layers!r}"
                lines.append(f"        self.{name} = nn.{recurrent_class}({constructor})")
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
        port_variables: dict[tuple[str, str], str] = {}
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            definition = item.definition_id
            name = item.variable_name
            source_vars = [
                port_variables[(binding.source_node_id, binding.source_port_id)]
                for binding in item.input_bindings
            ]
            source = source_vars[0] if source_vars else ""
            params = node.parameters
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None

            manifest = catalog.get(definition, node.definition_version)
            if manifest.operation is not None:
                lines.append(
                    f"        {name} = {emit_operation_source(manifest, 'pytorch', source_vars, params)}"
                )
                variables[item.node_id] = name
            elif manifest.source_kind == "runtime_input":
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
                variables[item.node_id] = name
            elif definition in {"input.constant", "input.parameter"}:
                lines.append(f"        {name} = self.{name}")
                variables[item.node_id] = name
            elif definition == "math.index":
                expression = self._index_expression(params.get("expression", ":"))
                lines.append(f"        {name} = {expression.replace('x', source, 1)}")
                variables[item.node_id] = name
            elif definition == "merge.average":
                lines.append(f"        {name} = torch.stack([{', '.join(source_vars)}], dim=0).sum(dim=0) / {len(source_vars)!r}")
                variables[item.node_id] = name
            elif definition == "merge.split":
                lines.append(
                    f"        {name} = torch.tensor_split({source}, {int(params.get('sections', 2))!r}, dim={int(params.get('axis', -1))!r})"
                )
                variables[item.node_id] = name
            elif definition == "output.detection":
                lines.append(f"        {name} = ({', '.join(source_vars)})")
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
            elif definition == "layer.prelu":
                alpha = f"self.{name}_alpha.reshape(self.{name}_alpha_broadcast)"
                lines.append(f"        {name} = torch.where({source} >= 0, {source}, {source} * {alpha})")
                variables[item.node_id] = name
            elif definition == "layer.rms_norm":
                lines.append(
                    f"        {name} = {source} / torch.sqrt(torch.mean({source} * {source}, dim=self.{name}_axis, keepdim=True) + self.{name}_epsilon)"
                )
                lines.append(
                    f"        {name} = {name} * self.{name}_scale.reshape(self.{name}_broadcast)"
                )
                variables[item.node_id] = name
            elif definition == "layer.unit_norm":
                lines.append(
                    f"        {name} = {source} / torch.linalg.vector_norm({source}, ord=2, dim=self.{name}_axis, keepdim=True).clamp_min(1e-12)"
                )
                variables[item.node_id] = name
            elif definition == "layer.group_norm":
                lines.append(f"        {name}_current = {source}.movedim(self.{name}_axis, 1)")
                lines.append(
                    f"        {name}_grouped = {name}_current.reshape({name}_current.shape[0], self.{name}_groups, -1)"
                )
                lines.append(f"        {name}_mean = {name}_grouped.mean(dim=2, keepdim=True)")
                lines.append(f"        {name}_variance = {name}_grouped.var(dim=2, keepdim=True, unbiased=False)")
                lines.append(
                    f"        {name}_current = (({name}_grouped - {name}_mean) / torch.sqrt({name}_variance + self.{name}_epsilon)).reshape({name}_current.shape)"
                )
                lines.append(
                    f"        if self.{name}_gamma is not None: {name}_current = {name}_current * self.{name}_gamma.reshape((1, -1) + (1,) * ({name}_current.dim() - 2))"
                )
                lines.append(
                    f"        if self.{name}_beta is not None: {name}_current = {name}_current + self.{name}_beta.reshape((1, -1) + (1,) * ({name}_current.dim() - 2))"
                )
                lines.append(f"        {name} = {name}_current.movedim(1, self.{name}_axis)")
                variables[item.node_id] = name
            elif definition == "layer.reshape":
                target_shape = tuple(int(value) for value in params.get("target_shape", [-1]))
                lines.append(
                    f"        {name} = torch.reshape({source}, ({source}.shape[0],) + {target_shape!r})"
                )
                variables[item.node_id] = name
            elif definition == "layer.permute":
                axes = tuple(int(value) for value in params.get("axes", [2, 1]))
                lines.append(f"        {name} = {source}.permute((0,) + {axes!r})")
                variables[item.node_id] = name
            elif definition == "layer.repeat":
                repeats = int(params.get("repeats", 2))
                axis = int(params.get("axis", 1))
                if axis == 1:
                    lines.append(
                        f"        {name} = {source}.unsqueeze(1).repeat(1, {repeats!r}, 1)"
                    )
                else:
                    lines.append(
                        f"        {name} = torch.repeat_interleave({source}, {repeats!r}, dim={axis!r})"
                    )
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
            elif definition in {"conv.conv3d", "conv.depthwise", "conv.separable"}:
                dimensions = 3 if definition == "conv.conv3d" else int(params.get("dimensions", 2))
                data_format = str(params.get("data_format", "channels_last"))
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                padding = str(params.get("padding", "same"))
                conv_input = f"{name}_input"
                lines.append(f"        {conv_input} = {source}")
                if data_format == "channels_last":
                    lines.append(
                        f"        {conv_input} = {conv_input}.permute((0, {dimensions + 1}, *range(1, {dimensions + 1}))).contiguous()"
                    )
                if padding == "same":
                    lines.append(
                        f"        {conv_input} = _same_pad_nd({conv_input}, {kernel_size!r}, {strides!r}, 0.0, {dilation!r})"
                    )
                if definition == "conv.separable":
                    lines.append(f"        {name} = self.{name}_pointwise(self.{name}_depthwise({conv_input}))")
                else:
                    lines.append(f"        {name} = self.{name}({conv_input})")
                if data_format == "channels_last":
                    lines.append(
                        f"        {name} = {name}.permute((0, *range(2, {dimensions + 2}), 1)).contiguous()"
                    )
                activation = params.get("activation", "linear")
                if activation != "linear":
                    lines.append(f"        {name} = ({self._activation(str(activation))})({name})")
                variables[item.node_id] = name
            elif definition == "conv.transposed":
                dimensions = int(params.get("dimensions", 2))
                data_format = str(params.get("data_format", "channels_last"))
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                padding = str(params.get("padding", "valid"))
                output_padding_value = params.get("output_padding")
                output_padding = (
                    (0,) * dimensions
                    if output_padding_value is None
                    else self._spatial_nonnegative(params, "output_padding", dimensions, [0] * dimensions)
                )
                conv_input = f"{name}_input"
                lines.append(f"        {conv_input} = {source}")
                if data_format == "channels_last":
                    lines.append(
                        f"        {conv_input} = {conv_input}.permute((0, {dimensions + 1}, *range(1, {dimensions + 1}))).contiguous()"
                    )
                lines.append(f"        {name} = self.{name}({conv_input})")
                if padding == "same":
                    lines.append(
                        f"        {name} = _crop_transposed_same({name}, {conv_input}.shape, {strides!r}, {output_padding!r})"
                    )
                if data_format == "channels_last":
                    lines.append(
                        f"        {name} = {name}.permute((0, *range(2, {dimensions + 2}), 1)).contiguous()"
                    )
                activation = params.get("activation", "linear")
                if activation != "linear":
                    lines.append(f"        {name} = ({self._activation(str(activation))})({name})")
                variables[item.node_id] = name
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
                variables[item.node_id] = name
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
                variables[item.node_id] = name
            elif definition == "conv.upsampling":
                dimensions = int(params.get("dimensions", 2))
                size = self._spatial_values(params, "size", dimensions, [2] * dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                interpolation = str(params.get("interpolation", "nearest"))
                lines.append(
                    f"        {name} = _upsample_nd({source}, {size!r}, {data_format!r}, {interpolation!r})"
                )
                variables[item.node_id] = name
            elif definition == "conv.padding":
                dimensions = int(params.get("dimensions", 2))
                pairs = self._spatial_pairs(params, "padding", dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                lines.append(f"        {name} = _pad_nd({source}, {pairs!r}, {data_format!r})")
                variables[item.node_id] = name
            elif definition == "conv.cropping":
                dimensions = int(params.get("dimensions", 2))
                pairs = self._spatial_pairs(params, "cropping", dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                current = f"{name}_current"
                lines.append(f"        {current} = {source}")
                if data_format == "channels_last":
                    lines.append(
                        f"        {current} = {current}.permute((0, {dimensions + 1}, *range(1, {dimensions + 1}))).contiguous()"
                    )
                slices = ["slice(None)"] * (dimensions + 2)
                for index, (before, after) in enumerate(pairs, start=2):
                    slices[index] = f"slice({before}, {-after if after else 'None'})"
                lines.append(f"        {current} = {current}[({', '.join(slices)},)]")
                if data_format == "channels_last":
                    lines.append(
                        f"        {name} = {current}.permute((0, *range(2, {dimensions + 2}), 1)).contiguous()"
                    )
                else:
                    lines.append(f"        {name} = {current}")
                variables[item.node_id] = name
            elif definition == "layer.activation":
                lines.append(
                    f"        {name} = ({self._activation(str(params.get('activation', 'relu')))})({source})"
                )
                variables[item.node_id] = name
            elif definition == "recurrent.simple_rnn":
                sequence_name = f"{name}_sequence"
                state_name = f"{name}_state"
                lines.append(f"        {sequence_name}, {state_name} = self.{name}({source})")
                if not params.get("return_sequences", False):
                    lines.append(f"        {sequence_name} = {sequence_name}[:, -1, :]")
                if params.get("return_state", False):
                    lines.append(f"        {name} = ({sequence_name}, {state_name}[-1])")
                else:
                    lines.append(f"        {name} = {sequence_name}")
            elif definition == "recurrent.bidirectional":
                sequence_name = f"{name}_sequence"
                lines.append(f"        {sequence_name}, _ = self.{name}({source})")
                units = int(params.get("units", 64))
                merge_mode = str(params.get("merge_mode", "concat"))
                if merge_mode != "concat":
                    forward_name = f"{name}_forward"
                    backward_name = f"{name}_backward"
                    lines.append(f"        {forward_name} = {sequence_name}[..., :{units}]")
                    lines.append(f"        {backward_name} = {sequence_name}[..., {units}:]")
                    if merge_mode == "sum":
                        lines.append(f"        {sequence_name} = {forward_name} + {backward_name}")
                    elif merge_mode == "mul":
                        lines.append(f"        {sequence_name} = {forward_name} * {backward_name}")
                    elif merge_mode == "ave":
                        lines.append(
                            f"        {sequence_name} = ({forward_name} + {backward_name}) / 2.0"
                        )
                if not params.get("return_sequences", False):
                    lines.append(f"        {sequence_name} = {sequence_name}[:, -1, :]")
                lines.append(f"        {name} = {sequence_name}")
            elif definition == "recurrent.stack":
                lines.append(f"        {name}, _ = self.{name}({source})")
                if not params.get("return_sequences", False):
                    lines.append(f"        {name} = {name}[:, -1, :]")
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

            output_ports = [
                port
                for port in materialized_outputs(manifest, params)
                if not port.enabled_when
                or all(
                    params.get(key) in expected
                    if isinstance(expected, list)
                    else params.get(key) == expected
                    for key, expected in port.enabled_when.items()
                )
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
        optimizer_parameters = {
            parameter.id: optimizer_value.parameters.get(parameter.id, parameter.default)
            for parameter in optimizer_option.parameters
        }
        optimizer_kwargs = [
            f"        lr={optimizer_parameters['learning_rate']!r},",
        ]
        if optimizer_value.id == "adam":
            optimizer_kwargs.extend(
                [
                    f"        betas=({optimizer_parameters['beta_1']!r}, {optimizer_parameters['beta_2']!r}),",
                    f"        eps={optimizer_parameters['epsilon']!r},",
                ]
            )
        elif optimizer_value.id == "adamw":
            optimizer_kwargs.extend(
                [
                    f"        betas=({optimizer_parameters['beta_1']!r}, {optimizer_parameters['beta_2']!r}),",
                    f"        eps={optimizer_parameters['epsilon']!r},",
                    f"        weight_decay={optimizer_parameters['weight_decay']!r},",
                ]
            )
        elif optimizer_value.id == "sgd":
            optimizer_kwargs.extend(
                [
                    f"        momentum={optimizer_parameters['momentum']!r},",
                    f"        nesterov={optimizer_parameters['nesterov']!r},",
                ]
            )
        elif optimizer_value.id == "rmsprop":
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
                "    scores = (prediction[..., 0] if prediction.shape[-1] == 1 else prediction[..., 1]).reshape(-1)",
                "    labels = (target[..., 0] if target.shape[-1] == 1 else target.argmax(dim=-1).to(prediction.dtype)).reshape(-1)",
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
                f"    if not {bool(native.compiler.parameters.get('enabled', False))!r}:",
                "        return model",
                "    return torch.compile(",
                "        model,",
                f"        mode={native.compiler.parameters.get('mode', 'default')!r},",
                f"        fullgraph={bool(native.compiler.parameters.get('fullgraph', False))!r},",
                f"        dynamic={bool(native.compiler.parameters.get('dynamic', False))!r},",
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
                "def configured_losses() -> dict[str, object]:",
                f"    return {{output_id: _LOSS_ADAPTERS[loss_id] for output_id, loss_id in { {output_id: loss_id for output_id, (loss_id, _) in resolved_losses.items()}!r}.items()}}",
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
                f"        'optimizer': {{'id': {optimizer_value.id!r}, 'parameters': {dict(optimizer_value.parameters)!r}}},",
                f"        'losses': {loss_payload!r},",
                f"        'metrics': {metric_payload!r},",
                f"        'target_contracts': {target_contracts!r},",
                f"        'metric_target_contracts': {metric_target_contracts!r},",
                "        'training': 'not_run',",
                "    }",
                "",
                "",
                "def build_compilation(model: GeneratedModel) -> dict[str, object]:",
                "    configuration = configured_compilation()",
                "    configuration.update({",
                "        'optimizer_runtime': build_optimizer(model),",
                "        'loss_functions': configured_losses(),",
                "        'metric_functions': configured_metrics(),",
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
