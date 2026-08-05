from __future__ import annotations

from math import ceil, floor, prod
from typing import Any, cast

from common.catalogs.loader import NodeCatalog
from domain.enums import DataType
from domain.graph import ModelGraph, ShapeSummary
from domain.tensor import TensorSpec
from validation.graph_utils import incoming_edges, topological_order


class ShapeInferenceError(ValueError):
    pass


def _int_parameter(parameters: dict[str, Any], key: str, default: int | None = None) -> int:
    value = parameters.get(key, default)
    if not isinstance(value, int):
        raise ShapeInferenceError(f"Parameter '{key}' must be an integer")
    return value


def _spatial_parameter(
    parameters: dict[str, Any],
    key: str,
    dimensions: int,
    default: list[int],
) -> list[int]:
    value = parameters.get(key, default)
    items = cast(list[Any], value) if isinstance(value, list) else []
    if not isinstance(value, list) or len(items) != dimensions:
        raise ShapeInferenceError(
            f"Parameter '{key}' must contain exactly {dimensions} dimensions"
        )
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in items):
        raise ShapeInferenceError(f"Parameter '{key}' must contain positive integers")
    return [int(item) for item in items]


def _convolution_size(
    size: object,
    *,
    kernel: int,
    stride: int,
    dilation: int,
    padding: str,
    transposed: bool,
) -> int | None:
    if not isinstance(size, int):
        return None
    effective_kernel = dilation * (kernel - 1) + 1
    if transposed:
        if padding == "same":
            return size * stride
        return (size - 1) * stride + effective_kernel
    if padding == "same":
        return ceil(size / stride)
    result = floor((size - effective_kernel) / stride) + 1
    if result <= 0:
        raise ShapeInferenceError("Kernel and stride produce a non-positive output dimension")
    return result


def infer_shapes(graph: ModelGraph, catalog: NodeCatalog) -> list[ShapeSummary]:
    nodes = {node.id: node for node in graph.nodes}
    inbound = incoming_edges(graph)
    outputs_by_node: dict[str, dict[str, TensorSpec]] = {}
    summaries: list[ShapeSummary] = []

    for node_id in topological_order(graph):
        node = nodes[node_id]
        manifest = catalog.get(node.definition_id, node.definition_version)
        input_specs: list[TensorSpec] = []
        for edge in inbound.get(node_id, []):
            source_outputs = outputs_by_node.get(edge.source_node_id, {})
            source_spec = source_outputs.get(edge.source_port_id)
            if source_spec is not None:
                input_specs.append(source_spec)

        rule = manifest.shape_rule.kind
        parameters = node.parameters
        output: TensorSpec
        parameter_count: int | None = 0

        if rule == "input":
            shape = parameters.get("shape", [None, 1])
            if not isinstance(shape, list):
                raise ShapeInferenceError("Input shape must be a list")
            input_shape = cast(list[Any], shape)
            output = TensorSpec(
                shape=input_shape, dtype=parameters.get("dtype", DataType.FLOAT32)
            )
        elif rule == "identity":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            output = input_specs[0]
        elif rule == "dense":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            if source.rank < 1:
                raise ShapeInferenceError("Dense requires rank >= 1")
            units = _int_parameter(parameters, "units", 32)
            input_features = source.shape[-1]
            output = TensorSpec(shape=[*source.shape[:-1], units], dtype=source.dtype)
            parameter_count = (
                input_features * units + (units if parameters.get("use_bias", True) else 0)
                if isinstance(input_features, int)
                else None
            )
        elif rule == "flatten":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            batch = source.shape[0] if source.shape else None
            dimensions = source.shape[1:] if source.shape else []
            integer_dimensions = [item for item in dimensions if isinstance(item, int)]
            flat = (
                prod(integer_dimensions)
                if len(integer_dimensions) == len(dimensions)
                else None
            )
            output = TensorSpec(shape=[batch, flat], dtype=source.dtype)
        elif rule == "reshape":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            target_shape = parameters.get("target_shape", [-1])
            if not isinstance(target_shape, list):
                raise ShapeInferenceError("target_shape must be a list")
            target_dimensions = cast(list[Any], target_shape)
            if sum(dimension == -1 for dimension in target_dimensions) > 1:
                raise ShapeInferenceError("target_shape can contain at most one inferred dimension")
            if any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension == 0
                or dimension < -1
                for dimension in target_dimensions
            ):
                raise ShapeInferenceError(
                    "target_shape dimensions must be positive integers or one -1"
                )
            source = input_specs[0]
            validated_target_dimensions = [int(dimension) for dimension in target_dimensions]
            resolved_shape: list[int | None] = [
                None if dimension == -1 else dimension for dimension in validated_target_dimensions
            ]
            if -1 in validated_target_dimensions:
                source_dimensions = source.shape[1:]
                known_target = prod(
                    dimension for dimension in validated_target_dimensions if dimension != -1
                )
                integer_source_dimensions: list[int] = [
                    dimension for dimension in source_dimensions if isinstance(dimension, int)
                ]
                if len(integer_source_dimensions) == len(source_dimensions):
                    source_size = prod(integer_source_dimensions)
                    if source_size % known_target != 0:
                        raise ShapeInferenceError(
                            "target_shape is incompatible with the input element count"
                        )
                    resolved_shape[validated_target_dimensions.index(-1)] = source_size // known_target
            output = TensorSpec(
                shape=[source.shape[0], *resolved_shape],
                dtype=source.dtype,
            )
        elif rule == "conv_nd":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            dimensions = _int_parameter(manifest.shape_rule.config, "dimensions", 2)
            expected_rank = dimensions + 2
            if source.rank != expected_rank:
                raise ShapeInferenceError(
                    f"{manifest.display_name} expects rank {expected_rank}"
                )
            filters = _int_parameter(parameters, "filters", 32)
            data_format = str(parameters.get("data_format", "channels_last"))
            if data_format not in {"channels_last", "channels_first"}:
                raise ShapeInferenceError("data_format must be channels_last or channels_first")
            kernel = _spatial_parameter(
                parameters, "kernel_size", dimensions, [3] * dimensions
            )
            strides = _spatial_parameter(
                parameters, "strides", dimensions, [1] * dimensions
            )
            dilation = _spatial_parameter(
                parameters, "dilation_rate", dimensions, [1] * dimensions
            )
            padding = str(parameters.get("padding", "valid"))
            if padding not in {"same", "valid"}:
                raise ShapeInferenceError("padding must be 'same' or 'valid'")
            transposed = node.definition_id == "conv.transposed"
            shape = list(source.shape)
            channel_index = dimensions + 1 if data_format == "channels_last" else 1
            spatial_indices = (
                list(range(1, dimensions + 1))
                if data_format == "channels_last"
                else list(range(2, dimensions + 2))
            )
            input_channels = shape[channel_index]
            for index, kernel_size, stride, dilation_rate in zip(
                spatial_indices, kernel, strides, dilation, strict=True
            ):
                shape[index] = _convolution_size(
                    shape[index],
                    kernel=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                    padding=padding,
                    transposed=transposed,
                )
            shape[channel_index] = filters
            output = TensorSpec(shape=shape, dtype=source.dtype)
            kernel_elements = prod(kernel)
            parameter_count = (
                kernel_elements * input_channels * filters
                + (filters if parameters.get("use_bias", True) else 0)
                if isinstance(input_channels, int)
                else None
            )
        elif rule == "pool_nd":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            dimensions = _int_parameter(manifest.shape_rule.config, "dimensions", 2)
            expected_rank = dimensions + 2
            if source.rank != expected_rank:
                raise ShapeInferenceError(
                    f"{manifest.display_name} expects rank {expected_rank}"
                )
            pool_size = _spatial_parameter(
                parameters, "pool_size", dimensions, [2] * dimensions
            )
            strides = _spatial_parameter(
                parameters, "strides", dimensions, pool_size
            )
            padding = str(parameters.get("padding", "valid"))
            if padding not in {"same", "valid"}:
                raise ShapeInferenceError("padding must be 'same' or 'valid'")
            data_format = str(parameters.get("data_format", "channels_last"))
            spatial_indices = (
                list(range(1, dimensions + 1))
                if data_format == "channels_last"
                else list(range(2, dimensions + 2))
            )
            shape = list(source.shape)
            for index, kernel_size, stride in zip(
                spatial_indices, pool_size, strides, strict=True
            ):
                shape[index] = _convolution_size(
                    shape[index],
                    kernel=kernel_size,
                    stride=stride,
                    dilation=1,
                    padding=padding,
                    transposed=False,
                )
            output = TensorSpec(shape=shape, dtype=source.dtype)
            parameter_count = 0
        elif rule == "global_pool_nd":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            dimensions = _int_parameter(manifest.shape_rule.config, "dimensions", 2)
            expected_rank = dimensions + 2
            if source.rank != expected_rank:
                raise ShapeInferenceError(
                    f"{manifest.display_name} expects rank {expected_rank}"
                )
            data_format = str(parameters.get("data_format", "channels_last"))
            channel_index = dimensions + 1 if data_format == "channels_last" else 1
            output = TensorSpec(
                shape=[source.shape[0], source.shape[channel_index]],
                dtype=source.dtype,
            )
            parameter_count = 0
        elif rule == "recurrent":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            if source.rank != 3:
                raise ShapeInferenceError("Recurrent layers require rank 3: batch, time, features")
            units = _int_parameter(parameters, "units", 64)
            return_sequences = bool(parameters.get("return_sequences", False))
            output_shape = [source.shape[0], source.shape[1], units] if return_sequences else [source.shape[0], units]
            output = TensorSpec(shape=output_shape, dtype=source.dtype)
            input_features = source.shape[-1]
            multiplier = 4 if node.definition_id == "recurrent.lstm" else 3 if node.definition_id == "recurrent.gru" else 1
            parameter_count = (
                multiplier * (input_features * units + units * units + units)
                if isinstance(input_features, int)
                else None
            )
        elif rule == "embedding":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            dim = _int_parameter(parameters, "embedding_dim", 128)
            vocab = _int_parameter(parameters, "vocabulary_size", 10000)
            output = TensorSpec(shape=[*input_specs[0].shape, dim], dtype=DataType.FLOAT32)
            parameter_count = vocab * dim
        elif rule == "merge_same":
            if len(input_specs) < 2:
                raise ShapeInferenceError(f"{node_id} requires at least two inputs")
            if any(not input_specs[0].compatible_with(spec) for spec in input_specs[1:]):
                raise ShapeInferenceError("Merge inputs must have compatible shapes")
            output = input_specs[0]
        elif rule == "concatenate":
            if len(input_specs) < 2:
                raise ShapeInferenceError(f"{node_id} requires at least two inputs")
            rank = input_specs[0].rank
            if any(spec.rank != rank for spec in input_specs):
                raise ShapeInferenceError("Concatenate inputs must have equal rank")
            axis = _int_parameter(parameters, "axis", -1)
            normalized_axis = axis if axis >= 0 else rank + axis
            shape = list(input_specs[0].shape)
            sizes = [spec.shape[normalized_axis] for spec in input_specs]
            integer_sizes = [size for size in sizes if isinstance(size, int)]
            shape[normalized_axis] = (
                sum(integer_sizes) if len(integer_sizes) == len(sizes) else None
            )
            output = TensorSpec(shape=shape, dtype=input_specs[0].dtype)
        elif rule == "attention":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires at least one input")
            source = input_specs[0]
            if source.rank != 3:
                raise ShapeInferenceError("Attention requires rank 3: batch, sequence, embedding")
            output = source
            model_dim = source.shape[-1]
            heads = _int_parameter(parameters, "num_heads", 8)
            if isinstance(model_dim, int) and model_dim % heads != 0:
                raise ShapeInferenceError("Embedding dimension must be divisible by num_heads")
            parameter_count = 4 * model_dim * model_dim if isinstance(model_dim, int) else None
        elif rule in {"output", "math", "slice", "transpose", "normalization", "dropout"} or rule == "split":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            output = input_specs[0]
        else:
            if not input_specs:
                raise ShapeInferenceError(f"No input available for shape rule '{rule}'")
            output = input_specs[0]
            parameter_count = None

        output_ports = manifest.outputs or []
        outputs = {port.id: output for port in output_ports}
        outputs_by_node[node_id] = outputs
        summaries.append(ShapeSummary(node_id=node_id, outputs=outputs, parameter_count=parameter_count))

    return summaries
