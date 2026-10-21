# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

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
from exporters.node_emitters import NodeEmitterContext, registry_for_backend
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
        loss_ids_payload = {
            output_id: loss_id for output_id, (loss_id, _) in resolved_losses.items()
        }
        metric_ids_payload = {
            output_id: [metric_id for metric_id, _ in values]
            for output_id, values in resolved_metrics.items()
        }
        loss_parameters_payload = {
            output_id: dict(value.parameters)
            for output_id, value in loss_configurations.items()
        }
        metric_parameters_payload = {
            output_id: [dict(value.parameters) for value in values]
            for output_id, values in metric_configurations.items()
        }
        lines = [
            "from __future__ import annotations",
            "",
            "import functools",
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
            "def _sinusoidal_encoding(value):",
            "    positions = torch.arange(value.shape[1], device=value.device, dtype=value.dtype).unsqueeze(1)",
            "    indices = torch.arange(value.shape[2], device=value.device, dtype=value.dtype).unsqueeze(0)",
            "    dimension = value.shape[2]",
            "    rates = torch.pow(value.new_tensor(10000.0), (2.0 * torch.floor(indices / 2.0)) / dimension)",
            "    angles = positions / rates",
            "    even = indices.to(torch.int64).remainder(2).eq(0)",
            "    encoding = torch.where(even, torch.sin(angles), torch.cos(angles))",
            "    return value + encoding.unsqueeze(0)",
            "",
            "",
            "def _multi_head_attention(query, context, query_projection, key_projection, value_projection, output_projection, heads, key_dim, value_dim, output_shape):",
            "    query_projection = query_projection(query).reshape(query.shape[0], query.shape[1], heads, key_dim).transpose(1, 2)",
            "    key_projection = key_projection(context).reshape(context.shape[0], context.shape[1], heads, key_dim).transpose(1, 2)",
            "    value_projection = value_projection(context).reshape(context.shape[0], context.shape[1], heads, value_dim).transpose(1, 2)",
            "    scores = torch.matmul(query_projection, key_projection.transpose(-2, -1)) / (key_dim ** 0.5)",
            "    weights = torch.softmax(scores, dim=-1)",
            "    attended = torch.matmul(weights, value_projection).transpose(1, 2).reshape(query.shape[0], query.shape[1], heads * value_dim)",
            "    result = output_projection(attended)",
            "    if output_shape is not None: result = result.reshape(result.shape[0], result.shape[1], *output_shape)",
            "    return result",
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
            elif definition == "transformer.positional_encoding":
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"PyTorch positional encoding node {node.id} requires a rank-3 input")
                if str(params.get("mode", "sinusoidal")) == "learned":
                    sequence_length = self._static_dimension(source_spec, 1)
                    feature_dimension = self._static_dimension(source_spec, 2)
                    if sequence_length is None or feature_dimension is None:
                        raise ValueError(
                            f"PyTorch learned positional encoding node {node.id} requires static sequence and feature dimensions"
                        )
                    lines.append(
                        f"        self.{name}_embedding = nn.Parameter(torch.zeros({(sequence_length, feature_dimension)!r}))"
                        if bool(params.get("trainable", True))
                        else f"        self.register_buffer({name + '_embedding'!r}, torch.zeros({(sequence_length, feature_dimension)!r}))"
                    )
            elif definition in {"attention.multi_head", "attention.self", "attention.cross"}:
                query_spec = source_spec
                context_spec = source_spec
                for binding in item.input_bindings:
                    binding_spec = summaries.get(binding.source_node_id)
                    if binding.target_port_id == "query":
                        query_spec = binding_spec
                    elif binding.target_port_id == "context":
                        context_spec = binding_spec
                if query_spec is None or context_spec is None or query_spec.rank != 3 or context_spec.rank != 3:
                    raise ValueError(f"PyTorch attention node {node.id} requires rank-3 query and context inputs")
                query_dimension = self._static_dimension(query_spec, -1)
                context_dimension = self._static_dimension(context_spec, -1)
                if query_dimension is None or context_dimension is None:
                    raise ValueError(f"PyTorch attention node {node.id} requires known embedding dimensions")
                if int(params.get("layers", 1)) != 1:
                    raise ValueError("Native PyTorch attention supports layers=1 only")
                heads = int(params.get("num_heads", 8))
                key_dimension = int(params.get("key_dim", 64))
                value_dimension = int(params.get("value_dim", key_dimension) or key_dimension)
                raw_output_shape = params.get("output_shape")
                if raw_output_shape is None:
                    output_shape = None
                    output_dimension = query_dimension
                elif isinstance(raw_output_shape, int) and not isinstance(raw_output_shape, bool):
                    output_shape = (int(raw_output_shape),)
                    output_dimension = int(raw_output_shape)
                elif isinstance(raw_output_shape, list) and raw_output_shape:
                    output_shape = tuple(int(value) for value in raw_output_shape)
                    output_dimension = 1
                    for value in output_shape:
                        output_dimension *= value
                else:
                    raise ValueError(f"PyTorch attention node {node.id} has an invalid output_shape")
                bias = bool(params.get("use_bias", True))
                lines.extend(
                    [
                        f"        self.{name}_query = nn.Linear({query_dimension!r}, {heads * key_dimension!r}, bias={bias!r})",
                        f"        self.{name}_key = nn.Linear({context_dimension!r}, {heads * key_dimension!r}, bias={bias!r})",
                        f"        self.{name}_value = nn.Linear({context_dimension!r}, {heads * value_dimension!r}, bias={bias!r})",
                        f"        self.{name}_output = nn.Linear({heads * value_dimension!r}, {output_dimension!r}, bias={bias!r})",
                    ]
                )
            elif definition in {"transformer.encoder_block", "transformer.decoder_block", "transformer.stack"}:
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"PyTorch {definition} node {node.id} requires a rank-3 query input")
                if float(params.get("dropout", 0.0)) != 0.0:
                    raise ValueError("Native PyTorch transformer blocks support dropout=0 only")
                query_dimension = self._static_dimension(source_spec, -1)
                context_spec = source_spec
                for binding in item.input_bindings:
                    if binding.target_port_id == "context":
                        context_spec = summaries.get(binding.source_node_id)
                if query_dimension is None or (
                    definition == "transformer.decoder_block"
                    and (context_spec is None or context_spec.rank != 3)
                ):
                    raise ValueError(f"PyTorch {definition} node {node.id} requires known rank-3 dimensions")
                context_dimension = self._static_dimension(context_spec, -1)
                if context_dimension is None:
                    raise ValueError(f"PyTorch {definition} node {node.id} requires a known context dimension")
                layer_count = int(params.get("layers", 1))
                if layer_count < 1:
                    raise ValueError("Transformer block layers must be positive")
                heads = int(params.get("num_heads", 8))
                key_dimension = int(params.get("key_dim", 64))
                hidden_dimension = query_dimension * int(params.get("ffn_multiplier", 4))
                pre_norm = bool(params.get("pre_norm", True))
                for layer_index in range(layer_count):
                    prefix = f"{name}_layer{layer_index}"
                    attention_specs = [("self", query_dimension)]
                    if definition == "transformer.decoder_block":
                        attention_specs.append(("cross", context_dimension))
                    for attention_kind, attention_context_dimension in attention_specs:
                        lines.extend(
                            [
                                f"        self.{prefix}_{attention_kind}_query = nn.Linear({query_dimension!r}, {heads * key_dimension!r})",
                                f"        self.{prefix}_{attention_kind}_key = nn.Linear({attention_context_dimension!r}, {heads * key_dimension!r})",
                                f"        self.{prefix}_{attention_kind}_value = nn.Linear({attention_context_dimension!r}, {heads * key_dimension!r})",
                                f"        self.{prefix}_{attention_kind}_output = nn.Linear({heads * key_dimension!r}, {query_dimension!r})",
                            ]
                        )
                    lines.extend(
                        [
                            f"        self.{prefix}_expand = nn.Linear({query_dimension!r}, {hidden_dimension!r})",
                            f"        self.{prefix}_project = nn.Linear({hidden_dimension!r}, {query_dimension!r})",
                        ]
                    )
                    norm_names = (
                        ["self_norm", "cross_norm", "ffn_norm"]
                        if definition == "transformer.decoder_block" and pre_norm
                        else ["self_norm", "cross_norm", "output_norm"]
                        if definition == "transformer.decoder_block"
                        else ["attention_norm", "ffn_norm"]
                        if pre_norm
                        else ["attention_norm", "output_norm"]
                    )
                    for norm_name in norm_names:
                        lines.append(
                            f"        self.{prefix}_{norm_name} = nn.LayerNorm({query_dimension!r})"
                        )
            elif definition == "transformer.feed_forward":
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"PyTorch feed-forward node {node.id} requires a rank-3 input")
                if float(params.get("dropout", 0.0)) != 0.0:
                    raise ValueError("Native PyTorch feed-forward export supports dropout=0 only")
                input_dimension = self._static_dimension(source_spec, -1)
                if input_dimension is None:
                    raise ValueError(f"PyTorch feed-forward node {node.id} requires a known embedding dimension")
                hidden_dimension = input_dimension * int(params.get("ffn_multiplier", 4))
                lines.extend(
                    [
                        f"        self.{name}_expand = nn.Linear({input_dimension!r}, {hidden_dimension!r})",
                        f"        self.{name}_project = nn.Linear({hidden_dimension!r}, {input_dimension!r})",
                    ]
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
        port_variables: dict[tuple[str, str], str] = {}
        emitter_registry = registry_for_backend("pytorch")
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            definition = item.definition_id
            name = item.variable_name
            source_vars = [
                port_variables[(binding.source_node_id, binding.source_port_id)]
                for binding in item.input_bindings
            ]
            source = source_vars[0] if source_vars else ""
            source_by_port = {
                binding.target_port_id: port_variables[
                    (binding.source_node_id, binding.source_port_id)
                ]
                for binding in item.input_bindings
            }
            params = node.parameters
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None

            manifest = catalog.get(definition, node.definition_version)
            mapping = manifest.implementations.get("pytorch")
            if (
                mapping is not None
                and emitter_registry.has_authoritative(mapping.adapter)
            ):
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
                if not output_ports:
                    raise ValueError(
                        f"No enabled PyTorch output port registered for {definition}"
                    )
                input_expressions = {
                    f"{binding.target_port_id}:{index}": port_variables[
                        (binding.source_node_id, binding.source_port_id)
                    ]
                    for index, binding in enumerate(item.input_bindings)
                }
                emission = emitter_registry.resolve(mapping.adapter)(
                    NodeEmitterContext(
                        graph=graph,
                        canonical_ir_node=item,
                        manifest=manifest,
                        implementation_mapping=mapping,
                        normalized_parameters=params,
                        input_expressions_by_target_port=input_expressions,
                        materialized_output_ports=tuple(output_ports),
                    )
                )
                lines.extend(f"        {statement}" for statement in emission.setup_statements)
                expressions: list[str] = []
                for port in output_ports:
                    expression = emission.output_expressions.get(port.id)
                    if expression is None:
                        raise ValueError(
                            f"PyTorch adapter {mapping.adapter} emitted an incomplete output set"
                        )
                    expressions.append(expression)
                if len(expressions) == 1:
                    lines.append(f"        {name} = {expressions[0]}")
                else:
                    lines.append(f"        {name} = ({', '.join(expressions)},)")
                variables[item.node_id] = name
            elif manifest.operation is not None:
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
            elif definition in {"transformer.encoder_block", "transformer.decoder_block", "transformer.stack"}:
                query = source_by_port.get("query", source)
                context = source_by_port.get("context", query)
                current = query
                layer_count = int(params.get("layers", 1))
                heads = int(params.get("num_heads", 8))
                key_dimension = int(params.get("key_dim", 64))
                activation = str(params.get("activation", "gelu"))
                pre_norm = bool(params.get("pre_norm", True))
                for layer_index in range(layer_count):
                    prefix = f"{name}_layer{layer_index}"
                    if definition in {"transformer.encoder_block", "transformer.stack"}:
                        attention_input = f"{prefix}_attention_input"
                        if pre_norm:
                            lines.append(f"        {attention_input} = self.{prefix}_attention_norm({current})")
                        else:
                            lines.append(f"        {attention_input} = {current}")
                        attention = f"{prefix}_attention"
                        lines.append(
                            f"        {attention} = _multi_head_attention({attention_input}, {attention_input}, self.{prefix}_self_query, self.{prefix}_self_key, self.{prefix}_self_value, self.{prefix}_self_output, {heads}, {key_dimension}, {key_dimension}, None)"
                        )
                        attention_residual = f"{prefix}_attention_residual"
                        lines.append(f"        {attention_residual} = {current} + {attention}")
                        ffn_input = f"{prefix}_ffn_input"
                        norm_name = "ffn_norm" if pre_norm else "attention_norm"
                        lines.append(f"        {ffn_input} = self.{prefix}_{norm_name}({attention_residual})")
                        current_cross = attention_residual
                    else:
                        self_input = f"{prefix}_self_input"
                        if pre_norm:
                            lines.append(f"        {self_input} = self.{prefix}_self_norm({current})")
                        else:
                            lines.append(f"        {self_input} = {current}")
                        self_attention = f"{prefix}_self_attention"
                        lines.append(
                            f"        {self_attention} = _multi_head_attention({self_input}, {self_input}, self.{prefix}_self_query, self.{prefix}_self_key, self.{prefix}_self_value, self.{prefix}_self_output, {heads}, {key_dimension}, {key_dimension}, None)"
                        )
                        self_residual = f"{prefix}_self_residual"
                        lines.append(f"        {self_residual} = {current} + {self_attention}")
                        current_self = f"{prefix}_current_self"
                        if pre_norm:
                            lines.append(f"        {current_self} = {self_residual}")
                        else:
                            lines.append(f"        {current_self} = self.{prefix}_self_norm({self_residual})")
                        cross_input = f"{prefix}_cross_input"
                        if pre_norm:
                            lines.append(f"        {cross_input} = self.{prefix}_cross_norm({current_self})")
                        else:
                            lines.append(f"        {cross_input} = {current_self}")
                        cross_attention = f"{prefix}_cross_attention"
                        lines.append(
                            f"        {cross_attention} = _multi_head_attention({cross_input}, {context}, self.{prefix}_cross_query, self.{prefix}_cross_key, self.{prefix}_cross_value, self.{prefix}_cross_output, {heads}, {key_dimension}, {key_dimension}, None)"
                        )
                        cross_residual = f"{prefix}_cross_residual"
                        lines.append(f"        {cross_residual} = {current_self} + {cross_attention}")
                        current_cross = f"{prefix}_current_cross"
                        if pre_norm:
                            lines.append(f"        {current_cross} = {cross_residual}")
                        else:
                            lines.append(f"        {current_cross} = self.{prefix}_cross_norm({cross_residual})")
                        ffn_input = f"{prefix}_ffn_input"
                        if pre_norm:
                            lines.append(f"        {ffn_input} = self.{prefix}_ffn_norm({current_cross})")
                        else:
                            lines.append(f"        {ffn_input} = {current_cross}")
                    lines.append(f"        {prefix}_feed_forward = self.{prefix}_expand({ffn_input})")
                    if activation != "linear":
                        lines.append(f"        {prefix}_feed_forward = ({self._activation(activation)})({prefix}_feed_forward)")
                    lines.append(f"        {prefix}_feed_forward = self.{prefix}_project({prefix}_feed_forward)")
                    current = f"{prefix}_output"
                    lines.append(f"        {current} = {current_cross} + {prefix}_feed_forward")
                    if not pre_norm:
                        lines.append(f"        {current} = self.{prefix}_output_norm({current})")
                lines.append(f"        {name} = {current}")
            elif definition in {"attention.multi_head", "attention.self", "attention.cross"}:
                if int(params.get("layers", 1)) != 1:
                    raise ValueError("Native PyTorch attention supports layers=1 only")
                raw_output_shape = params.get("output_shape")
                if raw_output_shape is None:
                    output_shape = None
                elif isinstance(raw_output_shape, int) and not isinstance(raw_output_shape, bool):
                    output_shape = (int(raw_output_shape),)
                elif isinstance(raw_output_shape, (list, tuple)) and raw_output_shape:
                    output_shape = tuple(int(value) for value in raw_output_shape)
                else:
                    raise ValueError(f"PyTorch attention node {node.id} has an invalid output_shape")
                query = source_by_port.get("query", source)
                context = source_by_port.get("context", query)
                value_dimension = int(
                    params.get("value_dim", params.get("key_dim", 64))
                    or params.get("key_dim", 64)
                )
                lines.append(
                    f"        {name} = _multi_head_attention({query}, {context}, self.{name}_query, self.{name}_key, self.{name}_value, self.{name}_output, {int(params.get('num_heads', 8))}, {int(params.get('key_dim', 64))}, {value_dimension}, {output_shape!r})"
                )
            elif definition == "transformer.attention_mask":
                lines.append(f"        {name} = {source}")
            elif definition == "transformer.residual":
                lines.append(f"        {name} = sum([{', '.join(source_vars)}])")
            elif definition == "transformer.positional_encoding":
                if str(params.get("mode", "sinusoidal")) == "learned":
                    lines.append(f"        {name} = {source} + self.{name}_embedding.unsqueeze(0)")
                else:
                    lines.append(f"        {name} = _sinusoidal_encoding({source})")
            elif definition == "transformer.feed_forward":
                activation = str(params.get("activation", "gelu"))
                lines.append(f"        {name} = self.{name}_expand({source})")
                if activation != "linear":
                    lines.append(f"        {name} = ({self._activation(activation)})({name})")
                lines.append(f"        {name} = self.{name}_project({name})")
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
        parameter_mapping = optimizer_option.implementations["pytorch"].parameter_mapping
        mapped_parameters: dict[str, list[Any]] = {}
        for parameter in optimizer_option.parameters:
            target = parameter_mapping.get(parameter.id, parameter.id)
            mapped_parameters.setdefault(target, []).append(optimizer_parameters[parameter.id])
        optimizer_kwargs = []
        for target, values in mapped_parameters.items():
            value: Any = values[0] if len(values) == 1 else tuple(values)
            optimizer_kwargs.append(f"        {target}={value!r},")

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
                "def _loss_binary_crossentropy(prediction: Tensor, target: Tensor, from_logits: bool = False, label_smoothing: float = 0.0, axis: int = -1) -> Tensor:",
                "    expected = target.to(torch.float32)",
                "    expected = expected * (1.0 - label_smoothing) + 0.5 * label_smoothing",
                "    values = F.binary_cross_entropy_with_logits(prediction, expected, reduction='none') if from_logits else F.binary_cross_entropy(_safe_probabilities(prediction), expected, reduction='none')",
                "    return values.mean(dim=axis).mean() if values.ndim > 1 else values.mean()",
                "",
                "",
                "def _loss_categorical_crossentropy(prediction: Tensor, target: Tensor, from_logits: bool = False, label_smoothing: float = 0.0, axis: int = -1) -> Tensor:",
                "    expected = target.to(torch.float32)",
                "    classes = prediction.shape[axis]",
                "    expected = expected * (1.0 - label_smoothing) + label_smoothing / max(int(classes), 1)",
                "    log_values = F.log_softmax(prediction, dim=axis) if from_logits else _safe_probabilities(prediction).log()",
                "    return -(expected * log_values).sum(dim=axis).mean()",
                "",
                "",
                "def _loss_sparse_categorical_crossentropy(prediction: Tensor, target: Tensor, from_logits: bool = False, label_smoothing: float = 0.0, axis: int = -1, ignore_class: int | None = None) -> Tensor:",
                "    labels = target.to(torch.long)",
                "    log_values = F.log_softmax(prediction, dim=axis) if from_logits else _safe_probabilities(prediction).log()",
                "    if label_smoothing == 0.0:",
                "        return F.nll_loss(log_values.movedim(axis, -1).reshape(-1, log_values.shape[axis]), labels.reshape(-1), ignore_index=-100 if ignore_class is None else int(ignore_class))",
                "    expected = F.one_hot(labels, num_classes=prediction.shape[axis]).to(prediction.dtype)",
                "    expected = expected * (1.0 - label_smoothing) + label_smoothing / max(int(prediction.shape[axis]), 1)",
                "    values = -(expected * log_values.movedim(axis, -1)).sum(dim=-1)",
                "    if ignore_class is not None:",
                "        values = values.masked_select(labels != int(ignore_class))",
                "    return values.mean()",
                "",
                "",
                "def _loss_huber(prediction: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:",
                "    return F.huber_loss(prediction, target.to(prediction.dtype), delta=delta)",
                "",
                "",
                "def _loss_kullback_leibler_divergence(prediction: Tensor, target: Tensor) -> Tensor:",
                "    probabilities = _safe_probabilities(prediction)",
                "    distribution = target.to(torch.float32).clamp_min(1e-7)",
                "    return F.kl_div(probabilities.log(), distribution, reduction='batchmean')",
                "",
                "",
                "def _loss_log_cosh(prediction: Tensor, target: Tensor) -> Tensor:",
                "    difference = (prediction - target.to(prediction.dtype)).abs()",
                "    return (F.softplus(2.0 * difference) - difference - torch.log(prediction.new_tensor(2.0))).mean()",
                "",
                "",
                "def _loss_mean_absolute_percentage_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    expected = target.to(prediction.dtype)",
                "    return ((prediction - expected).abs() / expected.abs().clamp_min(1e-7)).mean() * 100.0",
                "",
                "",
                "def _loss_mean_squared_logarithmic_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    expected = target.to(prediction.dtype).clamp_min(0.0)",
                "    return F.mse_loss(torch.log1p(prediction.clamp_min(0.0)), torch.log1p(expected))",
                "",
                "",
                "def _loss_hinge(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.relu(1.0 - target.to(prediction.dtype) * prediction).mean()",
                "",
                "",
                "def _loss_squared_hinge(prediction: Tensor, target: Tensor) -> Tensor:",
                "    return F.relu(1.0 - target.to(prediction.dtype) * prediction).square().mean()",
                "",
                "",
                "def _loss_categorical_hinge(prediction: Tensor, target: Tensor) -> Tensor:",
                "    expected = target.to(prediction.dtype)",
                "    positive = (prediction * expected).sum(dim=-1)",
                "    negative = ((1.0 - expected) * prediction).max(dim=-1).values",
                "    return F.relu(negative - positive + 1.0).mean()",
                "",
                "",
                "def _loss_poisson(prediction: Tensor, target: Tensor) -> Tensor:",
                "    expected = target.to(prediction.dtype)",
                "    rate = prediction.clamp_min(1e-7)",
                "    return (rate - expected * rate.log()).mean()",
                "",
                "",
                "def _loss_dice(prediction: Tensor, target: Tensor, axis: int | None = None) -> Tensor:",
                "    expected = target.to(prediction.dtype)",
                "    intersection = (prediction * expected).sum(dim=axis)",
                "    denominator = prediction.sum(dim=axis) + expected.sum(dim=axis)",
                "    return (1.0 - 2.0 * intersection / denominator.clamp_min(1e-7)).mean()",
                "",
                "",
                "def _loss_tversky(prediction: Tensor, target: Tensor, alpha: float = 0.5, beta: float = 0.5, axis: int | None = None) -> Tensor:",
                "    expected = target.to(prediction.dtype)",
                "    true_positive = (prediction * expected).sum(dim=axis)",
                "    false_positive = (prediction * (1.0 - expected)).sum(dim=axis)",
                "    false_negative = ((1.0 - prediction) * expected).sum(dim=axis)",
                "    denominator = true_positive + alpha * false_positive + beta * false_negative",
                "    return (1.0 - true_positive / denominator.clamp_min(1e-7)).mean()",
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
                "def _metric_top_k_categorical_accuracy(prediction: Tensor, target: Tensor, k: int = 5) -> Tensor:",
                "    limit = min(max(int(k), 1), int(prediction.shape[-1]))",
                "    expected = target.argmax(dim=-1)",
                "    candidates = prediction.topk(limit, dim=-1).indices",
                "    return candidates.eq(expected.unsqueeze(-1)).any(dim=-1).to(torch.float32).mean()",
                "",
                "",
                "def _metric_sparse_top_k_categorical_accuracy(prediction: Tensor, target: Tensor, k: int = 5) -> Tensor:",
                "    limit = min(max(int(k), 1), int(prediction.shape[-1]))",
                "    candidates = prediction.topk(limit, dim=-1).indices",
                "    return candidates.eq(target.to(torch.long).unsqueeze(-1)).any(dim=-1).to(torch.float32).mean()",
                "",
                "",
                "def _binary_labels(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> tuple[Tensor, Tensor]:",
                "    scores = prediction[..., 0] if prediction.shape[-1] == 1 else prediction",
                "    expected = target[..., 0] if target.shape[-1] == 1 else target",
                "    return (scores >= threshold).to(torch.long).reshape(-1), (expected >= 0.5).to(torch.long).reshape(-1)",
                "",
                "",
                "def _metric_true_positives(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:",
                "    predicted, expected = _binary_labels(prediction, target, threshold)",
                "    return ((predicted == 1) & (expected == 1)).to(torch.float32).sum()",
                "",
                "",
                "def _metric_true_negatives(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:",
                "    predicted, expected = _binary_labels(prediction, target, threshold)",
                "    return ((predicted == 0) & (expected == 0)).to(torch.float32).sum()",
                "",
                "",
                "def _metric_false_positives(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:",
                "    predicted, expected = _binary_labels(prediction, target, threshold)",
                "    return ((predicted == 1) & (expected == 0)).to(torch.float32).sum()",
                "",
                "",
                "def _metric_false_negatives(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:",
                "    predicted, expected = _binary_labels(prediction, target, threshold)",
                "    return ((predicted == 0) & (expected == 1)).to(torch.float32).sum()",
                "",
                "",
                "def _metric_classification_labels(prediction: Tensor, target: Tensor, threshold: float = 0.5, top_k: int | None = None, class_id: int | None = None) -> tuple[Tensor, Tensor]:",
                "    if class_id is not None:",
                "        scores = prediction[..., class_id] if prediction.shape[-1] > 1 else prediction[..., 0]",
                "        if target.shape == prediction.shape:",
                "            expected = target[..., class_id] if prediction.shape[-1] > 1 else target[..., 0]",
                "        else:",
                "            expected = (target.to(torch.long) == class_id).to(prediction.dtype)",
                "        return (scores >= threshold).to(torch.long).reshape(-1), (expected >= 0.5).to(torch.long).reshape(-1)",
                "    if top_k is not None and prediction.shape[-1] > 1:",
                "        limit = min(max(int(top_k), 1), int(prediction.shape[-1]))",
                "        candidates = prediction.topk(limit, dim=-1).indices",
                "        if target.shape == prediction.shape:",
                "            expected = target >= 0.5",
                "        else:",
                "            expected = torch.zeros_like(prediction, dtype=torch.bool).scatter_(-1, target.to(torch.long).unsqueeze(-1), True)",
                "        predicted = torch.zeros_like(prediction, dtype=torch.bool).scatter_(-1, candidates, True)",
                "        return predicted.reshape(-1).to(torch.long), expected.reshape(-1).to(torch.long)",
                "    if prediction.shape[-1] == 1:",
                "        scores = prediction[..., 0]",
                "        expected = target[..., 0] if target.shape == prediction.shape else target",
                "        return (scores >= threshold).to(torch.long).reshape(-1), (expected >= 0.5).to(torch.long).reshape(-1)",
                "    return _classification_labels(prediction, target)",
                "",
                "",
                "def _metric_precision(prediction: Tensor, target: Tensor, threshold: float = 0.5, top_k: int | None = None, class_id: int | None = None) -> Tensor:",
                "    predicted, expected = _metric_classification_labels(prediction, target, threshold, top_k, class_id)",
                "    true_positive = ((predicted == 1) & (expected == 1)).to(torch.float32).sum()",
                "    predicted_positive = (predicted == 1).to(torch.float32).sum()",
                "    return true_positive / predicted_positive.clamp_min(1.0)",
                "",
                "",
                "def _metric_recall(prediction: Tensor, target: Tensor, threshold: float = 0.5, top_k: int | None = None, class_id: int | None = None) -> Tensor:",
                "    predicted, expected = _metric_classification_labels(prediction, target, threshold, top_k, class_id)",
                "    true_positive = ((predicted == 1) & (expected == 1)).to(torch.float32).sum()",
                "    actual_positive = (expected == 1).to(torch.float32).sum()",
                "    return true_positive / actual_positive.clamp_min(1.0)",
                "",
                "",
                "def _auc_binary(scores: Tensor, labels: Tensor, thresholds: Tensor, curve: str, summation_method: str) -> Tensor:",
                "    expected = labels >= 0.5",
                "    positive_count = expected.to(torch.float32).sum()",
                "    negative_count = (~expected).to(torch.float32).sum()",
                "    if positive_count.item() == 0 or negative_count.item() == 0:",
                "        return scores.new_tensor(0.5)",
                "    x_values = []",
                "    y_values = []",
                "    for threshold in thresholds:",
                "        predicted = scores >= threshold",
                "        true_positive = (predicted & expected).to(torch.float32).sum()",
                "        false_positive = (predicted & ~expected).to(torch.float32).sum()",
                "        if curve == 'PR':",
                "            precision = true_positive / predicted.to(torch.float32).sum().clamp_min(1.0)",
                "            recall = true_positive / positive_count.clamp_min(1.0)",
                "            x_values.append(recall)",
                "            y_values.append(precision)",
                "        else:",
                "            x_values.append(false_positive / negative_count.clamp_min(1.0))",
                "            y_values.append(true_positive / positive_count.clamp_min(1.0))",
                "    x = torch.stack(x_values)",
                "    y = torch.stack(y_values)",
                "    order = torch.argsort(x)",
                "    x = x[order]",
                "    y = y[order]",
                "    widths = (x[1:] - x[:-1]).abs()",
                "    lower = torch.minimum(y[1:], y[:-1])",
                "    upper = torch.maximum(y[1:], y[:-1])",
                "    if summation_method == 'minoring':",
                "        return (lower * widths).sum().clamp(0.0, 1.0)",
                "    if summation_method == 'majoring':",
                "        return (upper * widths).sum().clamp(0.0, 1.0)",
                "    return (((y[1:] + y[:-1]) * 0.5) * widths).sum().clamp(0.0, 1.0)",
                "",
                "",
                "def _metric_auc(prediction: Tensor, target: Tensor, num_thresholds: int = 200, curve: str = 'ROC', summation_method: str = 'interpolation', thresholds: list[float] | None = None, multi_label: bool = False, num_labels: int | None = None, label_weights: list[float] | None = None, from_logits: bool = False) -> Tensor:",
                "    scores = prediction",
                "    if from_logits:",
                "        scores = scores.sigmoid()",
                "    if multi_label:",
                "        if num_labels is not None and scores.shape[-1] != num_labels:",
                "            scores = scores.reshape(-1, num_labels)",
                "            target = target.reshape(-1, num_labels)",
                "        else:",
                "            scores = scores.reshape(-1, scores.shape[-1])",
                "            target = target.reshape(-1, target.shape[-1])",
                "        results = []",
                "        threshold_values = torch.as_tensor(thresholds, dtype=scores.dtype, device=scores.device) if thresholds else torch.linspace(0.0, 1.0, max(int(num_thresholds), 2), dtype=scores.dtype, device=scores.device)",
                "        for index in range(scores.shape[-1]):",
                "            results.append(_auc_binary(scores[:, index], target[:, index], threshold_values, curve, summation_method))",
                "        values = torch.stack(results)",
                "        if label_weights:",
                "            weights = torch.as_tensor(label_weights, dtype=values.dtype, device=values.device)",
                "            return (values * weights[:values.shape[0]]).sum() / weights[:values.shape[0]].sum().clamp_min(1e-7)",
                "        return values.mean()",
                "    scores = scores[..., 0] if scores.shape[-1] == 1 else scores[..., 1]",
                "    labels = target[..., 0] if target.shape[-1] == 1 else (target[..., 1] if target.shape == prediction.shape else target)",
                "    threshold_values = torch.as_tensor(thresholds, dtype=scores.dtype, device=scores.device) if thresholds else torch.linspace(0.0, 1.0, max(int(num_thresholds), 2), dtype=scores.dtype, device=scores.device)",
                "    return _auc_binary(scores.reshape(-1), labels.reshape(-1), threshold_values, curve, summation_method)",
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
                "",
                "",
                "def _metric_mean_absolute_percentage_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    expected = target.to(prediction.dtype)",
                "    return ((prediction - expected).abs() / expected.abs().clamp_min(1e-7)).mean() * 100.0",
                "",
                "",
                "def _metric_mean_squared_logarithmic_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    expected = target.to(prediction.dtype).clamp_min(0.0)",
                "    return F.mse_loss(torch.log1p(prediction.clamp_min(0.0)), torch.log1p(expected))",
                "",
                "",
                "def _metric_cosine_similarity(prediction: Tensor, target: Tensor, axis: int = -1) -> Tensor:",
                "    return F.cosine_similarity(prediction, target.to(prediction.dtype), dim=axis).mean()",
                "",
                "",
                "def _metric_log_cosh_error(prediction: Tensor, target: Tensor) -> Tensor:",
                "    difference = (prediction - target.to(prediction.dtype)).abs()",
                "    return (F.softplus(2.0 * difference) - difference - torch.log(prediction.new_tensor(2.0))).mean()",
                "",
                "",
                "def _metric_prediction_mean(prediction: Tensor, target: Tensor) -> Tensor:",
                "    del target",
                "    return prediction.to(torch.float32).mean()",
                "",
                "",
                "def _metric_prediction_sum(prediction: Tensor, target: Tensor) -> Tensor:",
                "    del target",
                "    return prediction.to(torch.float32).sum()",
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
                "    'log_cosh': _loss_log_cosh,",
                "    'mean_absolute_percentage_error': _loss_mean_absolute_percentage_error,",
                "    'mean_squared_logarithmic_error': _loss_mean_squared_logarithmic_error,",
                "    'hinge': _loss_hinge,",
                "    'squared_hinge': _loss_squared_hinge,",
                "    'categorical_hinge': _loss_categorical_hinge,",
                "    'poisson': _loss_poisson,",
                "    'dice': _loss_dice,",
                "    'tversky': _loss_tversky,",
                *custom_loss_entries,
                "}",
                "",
                "",
                f"_LOSS_PARAMETERS = {loss_parameters_payload!r}",
                f"_METRIC_PARAMETERS = {metric_parameters_payload!r}",
                "",
                "",
                "_METRIC_ADAPTERS = {",
                "    'accuracy': _metric_accuracy,",
                "    'binary_accuracy': _metric_binary_accuracy,",
                "    'categorical_accuracy': _metric_categorical_accuracy,",
                "    'sparse_categorical_accuracy': _metric_sparse_categorical_accuracy,",
                "    'top_k_categorical_accuracy': _metric_top_k_categorical_accuracy,",
                "    'sparse_top_k_categorical_accuracy': _metric_sparse_top_k_categorical_accuracy,",
                "    'true_positives': _metric_true_positives,",
                "    'true_negatives': _metric_true_negatives,",
                "    'false_positives': _metric_false_positives,",
                "    'false_negatives': _metric_false_negatives,",
                "    'precision': _metric_precision,",
                "    'recall': _metric_recall,",
                "    'auc': _metric_auc,",
                "    'mean_absolute_error': _metric_mean_absolute_error,",
                "    'mean_squared_error': _metric_mean_squared_error,",
                "    'root_mean_squared_error': _metric_root_mean_squared_error,",
                "    'mean_absolute_percentage_error': _metric_mean_absolute_percentage_error,",
                "    'mean_squared_logarithmic_error': _metric_mean_squared_logarithmic_error,",
                "    'cosine_similarity': _metric_cosine_similarity,",
                "    'log_cosh_error': _metric_log_cosh_error,",
                "    'prediction_mean': _metric_prediction_mean,",
                "    'prediction_sum': _metric_prediction_sum,",
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
                f"    return {{output_id: functools.partial(_LOSS_ADAPTERS[loss_id], **_LOSS_PARAMETERS.get(output_id, {{}})) for output_id, loss_id in {loss_ids_payload!r}.items()}}",
                "",
                "",
                "def configured_metrics() -> dict[str, dict[str, object]]:",
                f"    return {{output_id: {{metric_id: functools.partial(_METRIC_ADAPTERS[metric_id], **parameters) for metric_id, parameters in zip(metric_ids, _METRIC_PARAMETERS.get(output_id, []), strict=True)}} for output_id, metric_ids in {metric_ids_payload!r}.items()}}",
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
