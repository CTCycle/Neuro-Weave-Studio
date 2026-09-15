from __future__ import annotations

import ast
from math import ceil, floor, prod
from typing import Any, cast

from common.catalogs.loader import NodeCatalog
from domain.enums import DataType
from domain.graph import ModelGraph, ShapeSummary
from domain.tensor import Dimension, TensorSpec
from validation.graph_utils import ordered_incoming_edges, topological_order
from validation.safe_expression import SafeExpressionError, validate_safe_expression


###############################################################################
class ShapeInferenceError(ValueError):
    pass

###############################################################################
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


###############################################################################
def _enabled_output_ports(manifest: Any, parameters: dict[str, Any]) -> list[Any]:
    return [
        port
        for port in manifest.outputs
        if _condition_matches(port.enabled_when, parameters)
    ]

###############################################################################
def _int_parameter(parameters: dict[str, Any], key: str, default: int | None = None) -> int:
    value = parameters.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ShapeInferenceError(f"Parameter '{key}' must be an integer")
    return value

###############################################################################
def _spatial_parameter(
    parameters: dict[str, Any],
    key: str,
    dimensions: int,
    default: list[int],
) -> list[int]:
    value = parameters.get(key, default)
    if value is None:
        value = default
    items = cast(list[Any], value) if isinstance(value, list) else []
    if not isinstance(value, list) or len(items) != dimensions:
        raise ShapeInferenceError(f"Parameter '{key}' must contain exactly {dimensions} dimensions")
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in items):
        raise ShapeInferenceError(f"Parameter '{key}' must contain positive integers")
    return [int(item) for item in items]

###############################################################################
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

###############################################################################
def _optional_positive_int(
    parameters: dict[str, Any],
    key: str,
    default: int,
) -> int:
    value = parameters.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ShapeInferenceError(f"Parameter '{key}' must be a positive integer")
    return value

###############################################################################
def _normalize_axis(axis: int, rank: int, *, allow_endpoint: bool = False) -> int:
    upper_bound = rank if allow_endpoint else rank - 1
    normalized = axis if axis >= 0 else (rank + 1 + axis if allow_endpoint else rank + axis)
    if normalized < 0 or normalized > upper_bound:
        raise ShapeInferenceError(f"Axis {axis} is outside rank {rank}")
    return normalized

###############################################################################
def _valid_spatial_pair(item: Any) -> bool:
    if not isinstance(item, list):
        return False
    pair_values = cast(list[Any], item)
    return len(pair_values) == 2 and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in pair_values
    )

###############################################################################
def _spatial_transform(
    source: TensorSpec,
    parameters: dict[str, Any],
    key: str,
    operation: str,
) -> TensorSpec:
    raw = parameters.get(key)
    if not isinstance(raw, list) or not raw:
        raise ShapeInferenceError(
            f"Parameter '{key}' must contain non-negative [before, after] pairs"
        )
    raw_items = cast(list[Any], raw)
    if any(not _valid_spatial_pair(item) for item in raw_items):
        raise ShapeInferenceError(
            f"Parameter '{key}' must contain non-negative [before, after] pairs"
        )
    raw_pairs = cast(list[list[Any]], raw_items)
    spatial_rank = source.rank - 2
    if len(raw_pairs) != spatial_rank:
        raise ShapeInferenceError(f"Parameter '{key}' must contain {spatial_rank} spatial pairs")
    shape = list(source.shape)
    for index, raw_pair in enumerate(raw_pairs, start=1):
        pair = [int(value) for value in raw_pair]
        dimension = shape[index]
        if isinstance(dimension, int):
            if operation == "pad":
                shape[index] = dimension + pair[0] + pair[1]
            else:
                result = dimension - pair[0] - pair[1]
                if result <= 0:
                    raise ShapeInferenceError("Cropping produces a non-positive output dimension")
                shape[index] = result
        else:
            shape[index] = None
    return TensorSpec(shape=shape, dtype=source.dtype)

###############################################################################
def _upsample_shape(source: TensorSpec, parameters: dict[str, Any]) -> TensorSpec:
    size = parameters.get("size")
    if not isinstance(size, list) or not size:
        raise ShapeInferenceError("Parameter 'size' must contain positive integers")
    size_values = cast(list[Any], size)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in size_values
    ):
        raise ShapeInferenceError("Parameter 'size' must contain positive integers")
    spatial_rank = source.rank - 2
    if len(size_values) != spatial_rank:
        raise ShapeInferenceError(
            f"Parameter 'size' must contain {spatial_rank} spatial dimensions"
        )
    shape = list(source.shape)
    for index, raw_multiplier in enumerate(size_values, start=1):
        multiplier = int(raw_multiplier)
        dimension = shape[index]
        if isinstance(dimension, int):
            shape[index] = dimension * multiplier
        else:
            shape[index] = None
    return TensorSpec(shape=shape, dtype=source.dtype)

###############################################################################
def _attention_output_dimensions(
    parameters: dict[str, Any], query_dimension: int | None
) -> tuple[list[int | None], int | None]:
    raw_output_shape = parameters.get("output_shape")
    if raw_output_shape is None:
        if query_dimension is None:
            return [None], None
        return [query_dimension], query_dimension
    if isinstance(raw_output_shape, int) and not isinstance(raw_output_shape, bool):
        dimensions = [raw_output_shape]
    elif isinstance(raw_output_shape, list):
        dimensions = cast(list[Any], raw_output_shape)
    else:
        raise ShapeInferenceError("Attention output_shape must be a positive integer list")
    if not dimensions or any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        for dimension in dimensions
    ):
        raise ShapeInferenceError("Attention output_shape must contain positive integers")
    return [int(dimension) for dimension in dimensions], prod(dimensions)

###############################################################################
def _attention_parameter_count(
    *,
    query_dimension: int | None,
    context_dimension: int | None,
    heads: int,
    key_dimension: int,
    value_dimension: int,
    output_dimension: int | None,
    use_bias: bool,
) -> int | None:
    if query_dimension is None or context_dimension is None or output_dimension is None:
        return None
    projection_bias = heads * key_dimension + heads * key_dimension + heads * value_dimension
    output_bias = output_dimension
    return (
        query_dimension * heads * key_dimension
        + context_dimension * heads * key_dimension
        + context_dimension * heads * value_dimension
        + heads * value_dimension * output_dimension
        + (projection_bias + output_bias if use_bias else 0)
    )

###############################################################################
def _operation_dtype(source: TensorSpec, output_dtype: str, parameters: dict[str, Any]) -> DataType:
    if output_dtype == "bool":
        return DataType.BOOL
    if output_dtype == "int64":
        return DataType.INT64
    if output_dtype == "parameter":
        value = parameters.get("dtype", source.dtype)
        try:
            return DataType(value)
        except ValueError as exc:
            raise ShapeInferenceError(f"Unsupported operation dtype '{value}'") from exc
    return source.dtype

###############################################################################
def _slice_integer_literal(node: ast.AST) -> int | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        value = int(node.operand.value)
        return value if isinstance(node.op, ast.UAdd) else -value
    return None

###############################################################################
def _slice_expression_shape(source: TensorSpec, expression: str) -> TensorSpec:
    try:
        validate_safe_expression(expression)
        tree = ast.parse(expression, mode="eval")
    except (SafeExpressionError, SyntaxError) as exc:
        raise ShapeInferenceError(str(exc)) from exc
    if not isinstance(tree.body, ast.Subscript):
        return TensorSpec(shape=list(source.shape), dtype=source.dtype)
    slice_node = tree.body.slice
    items = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
    ellipsis_indices = [
        index
        for index, item in enumerate(items)
        if isinstance(item, ast.Constant) and item.value is Ellipsis
    ]
    if len(ellipsis_indices) > 1:
        raise ShapeInferenceError("A slice expression may contain at most one ellipsis")
    if ellipsis_indices:
        ellipsis_index = ellipsis_indices[0]
        fill_count = source.rank - (len(items) - 1)
        if fill_count < 0:
            raise ShapeInferenceError("Slice expression has more indices than the input rank")
        items = [
            *items[:ellipsis_index],
            *[ast.Slice() for _ in range(fill_count)],
            *items[ellipsis_index + 1 :],
        ]
    if len(items) > source.rank:
        raise ShapeInferenceError("Slice expression has more indices than the input rank")
    items.extend(ast.Slice() for _ in range(source.rank - len(items)))
    output_shape: list[Dimension] = []
    for dimension, item in zip(source.shape, items, strict=True):
        index = _slice_integer_literal(item)
        if index is not None:
            if isinstance(dimension, int) and not -dimension <= index < dimension:
                raise ShapeInferenceError("Slice index is outside the input dimension")
            continue
        if not isinstance(item, ast.Slice):
            raise ShapeInferenceError("Slice expression contains an unsupported index")
        step = _slice_integer_literal(item.step) if item.step is not None else None
        if step == 0:
            raise ShapeInferenceError("Slice step cannot be zero")
        if isinstance(dimension, int):
            lower = _slice_integer_literal(item.lower) if item.lower is not None else None
            upper = _slice_integer_literal(item.upper) if item.upper is not None else None
            start, stop, stride = slice(lower, upper, step).indices(dimension)
            output_shape.append(len(range(start, stop, stride)))
        else:
            output_shape.append(None)
    return TensorSpec(shape=output_shape, dtype=source.dtype)

###############################################################################
def _operation_shape(
    node_id: str,
    manifest: Any,
    input_specs: list[TensorSpec],
    parameters: dict[str, Any],
) -> TensorSpec:
    operation = manifest.operation
    if operation is None:
        raise ShapeInferenceError(f"{node_id} is not an operation manifest")
    if not input_specs:
        raise ShapeInferenceError(f"{node_id} requires an input")
    source = input_specs[0]
    family = operation.family
    if family == "unary":
        return TensorSpec(
            shape=list(source.shape),
            dtype=_operation_dtype(source, operation.output_dtype, parameters),
        )
    if family in {"binary", "elementwise"}:
        if len(input_specs) < 2:
            raise ShapeInferenceError(f"{node_id} requires at least two inputs")
        if operation.shape_behavior == "matmul":
            left, right = input_specs[0], input_specs[1]
            # Graph input shapes include the runtime batch dimension.  The
            # emitted backend operation therefore consumes batch-first matrix
            # tensors, not unbatched rank-2 matrices.
            if left.rank < 3 or right.rank < 3:
                raise ShapeInferenceError(
                    "Matrix multiplication requires rank-3 or higher batch-first inputs"
                )
            left_inner, right_inner = left.shape[-1], right.shape[-2]
            if (
                isinstance(left_inner, int)
                and isinstance(right_inner, int)
                and left_inner != right_inner
            ):
                raise ShapeInferenceError("Matrix multiplication inner dimensions must match")
            if left.shape[:-2] != right.shape[:-2] and any(
                isinstance(left_value, int)
                and isinstance(right_value, int)
                and left_value != right_value
                for left_value, right_value in zip(left.shape[:-2], right.shape[:-2], strict=False)
            ):
                raise ShapeInferenceError("Matrix multiplication batch dimensions must match")
            shape = [*left.shape[:-2], left.shape[-2], right.shape[-1]]
            return TensorSpec(
                shape=shape,
                dtype=_operation_dtype(left, operation.output_dtype, parameters),
            )
        if any(not source.compatible_with(spec) for spec in input_specs[1:]):
            raise ShapeInferenceError("Element-wise operation inputs must have compatible shapes")
        return TensorSpec(
            shape=list(source.shape),
            dtype=_operation_dtype(source, operation.output_dtype, parameters),
        )
    if family == "reduction":
        axis = _normalize_axis(_int_parameter(parameters, "axis", -1), source.rank)
        shape = list(source.shape)
        if bool(parameters.get("keepdims", False)):
            shape[axis] = 1
        else:
            shape = [dimension for index, dimension in enumerate(shape) if index != axis]
        return TensorSpec(
            shape=shape,
            dtype=_operation_dtype(source, operation.output_dtype, parameters),
        )

    behavior = operation.shape_behavior
    if behavior == "reshape":
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
            raise ShapeInferenceError("target_shape dimensions must be positive integers or one -1")
        resolved: list[int | None] = [
            None if dimension == -1 else int(dimension) for dimension in target_dimensions
        ]
        if -1 in target_dimensions:
            source_dimensions = source.shape[1:]
            known_source = [
                dimension for dimension in source_dimensions if isinstance(dimension, int)
            ]
            known_target = prod(dimension for dimension in target_dimensions if dimension != -1)
            if len(known_source) == len(source_dimensions):
                source_size = prod(known_source)
                if source_size % known_target != 0:
                    raise ShapeInferenceError(
                        "target_shape is incompatible with the input element count"
                    )
                resolved[target_dimensions.index(-1)] = source_size // known_target
        return TensorSpec(
            shape=[source.shape[0], *resolved],
            dtype=_operation_dtype(source, operation.output_dtype, parameters),
        )
    if behavior == "transpose":
        axes = parameters.get("axes", [])
        if axes == []:
            axes = list(reversed(range(source.rank)))
        if not isinstance(axes, list):
            raise ShapeInferenceError("Transpose axes must contain integers")
        axis_items = cast(list[Any], axes)
        if any(not isinstance(axis, int) or isinstance(axis, bool) for axis in axis_items):
            raise ShapeInferenceError("Transpose axes must contain integers")
        integer_axes = [int(axis) for axis in axis_items]
        if sorted(integer_axes) != list(range(source.rank)):
            raise ShapeInferenceError("Transpose axes must be a permutation of the input rank")
        return TensorSpec(shape=[source.shape[axis] for axis in integer_axes], dtype=source.dtype)
    if behavior == "expand_dims":
        axis = _normalize_axis(
            _int_parameter(parameters, "axis", -1), source.rank, allow_endpoint=True
        )
        shape = list(source.shape)
        shape.insert(axis, 1)
        return TensorSpec(shape=shape, dtype=source.dtype)
    if behavior == "squeeze":
        axis = _normalize_axis(_int_parameter(parameters, "axis", -1), source.rank)
        dimension = source.shape[axis]
        if isinstance(dimension, int) and dimension != 1:
            raise ShapeInferenceError("Squeeze axis must have dimension 1")
        return TensorSpec(
            shape=[value for index, value in enumerate(source.shape) if index != axis],
            dtype=source.dtype,
        )
    if behavior == "flatten":
        dimensions = source.shape[1:]
        known_dimensions = [dimension for dimension in dimensions if isinstance(dimension, int)]
        flat = prod(known_dimensions) if len(known_dimensions) == len(dimensions) else None
        return TensorSpec(shape=[source.shape[0], flat], dtype=source.dtype)
    if behavior == "repeat":
        axis = _normalize_axis(_int_parameter(parameters, "axis", -1), source.rank)
        repeats = _optional_positive_int(parameters, "repeats", 1)
        shape = list(source.shape)
        dimension = shape[axis]
        shape[axis] = int(dimension) * repeats if isinstance(dimension, int) else None
        return TensorSpec(shape=shape, dtype=source.dtype)
    if behavior == "tile":
        multiples = parameters.get("multiples", [])
        if not isinstance(multiples, list):
            raise ShapeInferenceError(f"multiples must contain exactly {source.rank} values")
        multiple_items = cast(list[Any], multiples)
        if len(multiple_items) != source.rank:
            raise ShapeInferenceError(f"multiples must contain exactly {source.rank} values")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in multiple_items
        ):
            raise ShapeInferenceError("multiples must contain positive integers")
        shape: list[Dimension] = [
            dimension * int(multiple) if isinstance(dimension, int) else None
            for dimension, multiple in zip(source.shape, multiple_items, strict=True)
        ]
        return TensorSpec(shape=shape, dtype=source.dtype)
    if behavior == "cast":
        return TensorSpec(
            shape=list(source.shape),
            dtype=_operation_dtype(source, operation.output_dtype, parameters),
        )
    if behavior in {"concatenate", "stack"}:
        if len(input_specs) < 2:
            raise ShapeInferenceError(f"{node_id} requires at least two inputs")
        rank = source.rank
        axis = _int_parameter(parameters, "axis", -1 if behavior == "concatenate" else 0)
        normalized_axis = _normalize_axis(axis, rank, allow_endpoint=behavior == "stack")
        if any(spec.rank != rank or spec.dtype != source.dtype for spec in input_specs[1:]):
            raise ShapeInferenceError("Tensor operation inputs must have equal rank and dtype")
        for index in range(rank):
            if index == normalized_axis:
                continue
            expected = source.shape[index]
            if any(
                isinstance(expected, int)
                and isinstance(spec.shape[index], int)
                and expected != spec.shape[index]
                for spec in input_specs[1:]
            ):
                raise ShapeInferenceError(
                    "Tensor operation inputs must match on non-concatenated dimensions"
                )
        shape = list(source.shape)
        if behavior == "stack":
            shape.insert(normalized_axis, len(input_specs))
        else:
            sizes = [spec.shape[normalized_axis] for spec in input_specs]
            shape[normalized_axis] = (
                sum(size for size in sizes if isinstance(size, int))
                if all(isinstance(size, int) for size in sizes)
                else None
            )
        return TensorSpec(shape=shape, dtype=source.dtype)
    if behavior == "take":
        axis = _normalize_axis(_int_parameter(parameters, "axis", -1), source.rank)
        return TensorSpec(
            shape=[value for index, value in enumerate(source.shape) if index != axis],
            dtype=source.dtype,
        )
    if behavior == "expression":
        expression = parameters.get("expression")
        if not isinstance(expression, str):
            raise ShapeInferenceError("Expression must be a string")
        return _slice_expression_shape(source, expression)
    raise ShapeInferenceError(f"No shape inference rule for operation behavior '{behavior}'")

###############################################################################
def infer_shapes(graph: ModelGraph, catalog: NodeCatalog) -> list[ShapeSummary]:
    nodes = {node.id: node for node in graph.nodes}
    outputs_by_node: dict[str, dict[str, TensorSpec]] = {}
    summaries: list[ShapeSummary] = []

    for node_id in topological_order(graph):
        node = nodes[node_id]
        manifest = catalog.get(node.definition_id, node.definition_version)
        input_specs: list[TensorSpec] = []
        input_specs_by_port: dict[str, TensorSpec] = {}
        for edge in ordered_incoming_edges(graph, node_id, manifest):
            source_outputs = outputs_by_node.get(edge.source_node_id, {})
            source_spec = source_outputs.get(edge.source_port_id)
            if source_spec is not None:
                input_specs.append(source_spec)
                input_specs_by_port[edge.target_port_id] = source_spec

        rule = manifest.shape_rule.kind
        parameters = node.parameters
        output: TensorSpec
        parameter_count: int | None = 0

        if manifest.operation is not None:
            output = _operation_shape(node_id, manifest, input_specs, parameters)
            parameter_count = 0
        elif rule == "input":
            shape = parameters.get("shape", [None, 1])
            if not isinstance(shape, list):
                raise ShapeInferenceError("Input shape must be a list")
            input_shape = cast(list[Any], shape)
            if not input_shape or input_shape[0] is not None:
                raise ShapeInferenceError(
                    "Input shape must include an unknown leading batch dimension"
                )
            output = TensorSpec(shape=input_shape, dtype=parameters.get("dtype", DataType.FLOAT32))
        elif rule == "identity":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            if (
                node.definition_id == "transformer.positional_encoding"
                and parameters.get("mode", "sinusoidal") != "sinusoidal"
            ):
                raise ShapeInferenceError(
                    "Only sinusoidal positional encoding is supported by the Keras runtime"
                )
            if node.definition_id == "conv.padding":
                output = _spatial_transform(source, parameters, "padding", "pad")
            elif node.definition_id == "conv.cropping":
                output = _spatial_transform(source, parameters, "cropping", "crop")
            elif node.definition_id == "conv.upsampling":
                output = _upsample_shape(source, parameters)
            else:
                output = source
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
            flat = prod(integer_dimensions) if len(integer_dimensions) == len(dimensions) else None
            output = TensorSpec(shape=[batch, flat], dtype=source.dtype)
        elif rule == "reshape":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            target_shape = parameters.get("target_shape", [-1])
            if not isinstance(target_shape, list):
                raise ShapeInferenceError("target_shape must be a list")
            target_dimensions = cast(list[Any], target_shape)
            if not target_dimensions:
                raise ShapeInferenceError("target_shape must contain at least one dimension")
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
                    resolved_shape[validated_target_dimensions.index(-1)] = (
                        source_size // known_target
                    )
            else:
                source_dimensions = source.shape[1:]
                integer_source_dimensions = [
                    dimension for dimension in source_dimensions if isinstance(dimension, int)
                ]
                if len(integer_source_dimensions) == len(source_dimensions):
                    source_size = prod(integer_source_dimensions)
                    target_size = prod(validated_target_dimensions)
                    if source_size != target_size:
                        raise ShapeInferenceError(
                            "target_shape is incompatible with the input element count"
                        )
            output = TensorSpec(
                shape=[source.shape[0], *resolved_shape],
                dtype=source.dtype,
            )
        elif rule == "conv_nd":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            dimensions = _int_parameter(
                parameters,
                "dimensions",
                _int_parameter(manifest.shape_rule.config, "dimensions", 2),
            )
            expected_rank = dimensions + 2
            if source.rank != expected_rank:
                raise ShapeInferenceError(f"{manifest.display_name} expects rank {expected_rank}")
            filters = _int_parameter(parameters, "filters", 32)
            data_format = str(parameters.get("data_format", "channels_last"))
            if data_format not in {"channels_last", "channels_first"}:
                raise ShapeInferenceError("data_format must be channels_last or channels_first")
            kernel = _spatial_parameter(parameters, "kernel_size", dimensions, [3] * dimensions)
            strides = _spatial_parameter(parameters, "strides", dimensions, [1] * dimensions)
            dilation = _spatial_parameter(parameters, "dilation_rate", dimensions, [1] * dimensions)
            if any(
                stride > 1 and dilation_rate > 1
                for stride, dilation_rate in zip(strides, dilation, strict=True)
            ):
                raise ShapeInferenceError(
                    "Convolution cannot use strides > 1 with dilation_rate > 1"
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
            groups = _optional_positive_int(parameters, "groups", 1)
            if isinstance(input_channels, int):
                if input_channels % groups != 0:
                    raise ShapeInferenceError("Input channels must be divisible by groups")
                if filters % groups != 0:
                    raise ShapeInferenceError("Filters must be divisible by groups")
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
                kernel_elements * (input_channels // groups) * filters
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
                raise ShapeInferenceError(f"{manifest.display_name} expects rank {expected_rank}")
            pool_size = _spatial_parameter(parameters, "pool_size", dimensions, [2] * dimensions)
            strides = _spatial_parameter(parameters, "strides", dimensions, pool_size)
            padding = str(parameters.get("padding", "valid"))
            if padding not in {"same", "valid"}:
                raise ShapeInferenceError("padding must be 'same' or 'valid'")
            data_format = str(parameters.get("data_format", "channels_last"))
            if data_format not in {"channels_last", "channels_first"}:
                raise ShapeInferenceError("data_format must be channels_last or channels_first")
            spatial_indices = (
                list(range(1, dimensions + 1))
                if data_format == "channels_last"
                else list(range(2, dimensions + 2))
            )
            shape = list(source.shape)
            for index, kernel_size, stride in zip(spatial_indices, pool_size, strides, strict=True):
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
                raise ShapeInferenceError(f"{manifest.display_name} expects rank {expected_rank}")
            data_format = str(parameters.get("data_format", "channels_last"))
            if data_format not in {"channels_last", "channels_first"}:
                raise ShapeInferenceError("data_format must be channels_last or channels_first")
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
            output_shape = (
                [source.shape[0], source.shape[1], units]
                if return_sequences
                else [source.shape[0], units]
            )
            output = TensorSpec(shape=output_shape, dtype=source.dtype)
            input_features = source.shape[-1]
            multiplier = (
                4
                if node.definition_id == "recurrent.lstm"
                else 3
                if node.definition_id == "recurrent.gru"
                else 1
            )
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
            if input_specs[0].dtype not in {DataType.INT32, DataType.INT64}:
                raise ShapeInferenceError("Embedding inputs must use int32 or int64 dtype")
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
            normalized_axis = _normalize_axis(
                axis,
                rank,
                allow_endpoint=node.definition_id == "merge.stack",
            )
            if any(spec.dtype != input_specs[0].dtype for spec in input_specs[1:]):
                raise ShapeInferenceError("Concatenate inputs must have the same dtype")
            for index in range(rank):
                if index == normalized_axis:
                    continue
                expected = input_specs[0].shape[index]
                for spec in input_specs[1:]:
                    actual = spec.shape[index]
                    if isinstance(expected, int) and isinstance(actual, int) and expected != actual:
                        raise ShapeInferenceError(
                            "Concatenate inputs must match on every non-concatenated dimension"
                        )
            shape = list(input_specs[0].shape)
            if node.definition_id == "merge.stack":
                output = TensorSpec(
                    shape=[*shape[:normalized_axis], len(input_specs), *shape[normalized_axis:]],
                    dtype=input_specs[0].dtype,
                )
            else:
                sizes = [spec.shape[normalized_axis] for spec in input_specs]
                integer_sizes = [size for size in sizes if isinstance(size, int)]
                shape[normalized_axis] = (
                    sum(integer_sizes) if len(integer_sizes) == len(sizes) else None
                )
                output = TensorSpec(shape=shape, dtype=input_specs[0].dtype)
        elif rule == "split":
            if len(input_specs) != 1:
                raise ShapeInferenceError(f"{node_id} requires exactly one input")
            source = input_specs[0]
            sections = _optional_positive_int(parameters, "sections", 2)
            axis = _int_parameter(parameters, "axis", -1)
            normalized_axis = _normalize_axis(axis, source.rank)
            shape = list(source.shape)
            dimension = shape[normalized_axis]
            if isinstance(dimension, int):
                if dimension % sections != 0:
                    raise ShapeInferenceError(
                        "Split sections must evenly divide the selected dimension"
                    )
                shape[normalized_axis] = dimension // sections
            else:
                shape[normalized_axis] = None
            output = TensorSpec(shape=shape, dtype=source.dtype)
        elif rule == "attention":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires at least one input")
            query = input_specs_by_port.get("query", input_specs[0])
            context = input_specs_by_port.get("context", query)
            if query.rank != 3 or context.rank != 3:
                raise ShapeInferenceError("Attention requires rank 3: batch, sequence, embedding")
            if (
                isinstance(query.shape[0], int)
                and isinstance(context.shape[0], int)
                and query.shape[0] != context.shape[0]
            ):
                raise ShapeInferenceError("Attention query and context batch dimensions must match")
            if node.definition_id in {
                "transformer.encoder_block",
                "transformer.stack",
                "transformer.decoder_block",
                "transformer.feed_forward",
            }:
                source_dimension = query.shape[-1]
                if not isinstance(source_dimension, int):
                    raise ShapeInferenceError(
                        "Transformer blocks require a known embedding dimension"
                    )
                multiplier = _optional_positive_int(parameters, "ffn_multiplier", 4)
                hidden_dimension = source_dimension * multiplier
                feed_forward_count = (
                    source_dimension * hidden_dimension
                    + hidden_dimension
                    + hidden_dimension * source_dimension
                    + source_dimension
                )
                if node.definition_id == "transformer.feed_forward":
                    parameter_count = feed_forward_count
                else:
                    heads = _optional_positive_int(parameters, "num_heads", 8)
                    key_dimension = _optional_positive_int(parameters, "key_dim", 64)
                    layer_count = _optional_positive_int(parameters, "layers", 1)
                    self_attention_count = _attention_parameter_count(
                        query_dimension=source_dimension,
                        context_dimension=source_dimension,
                        heads=heads,
                        key_dimension=key_dimension,
                        value_dimension=key_dimension,
                        output_dimension=source_dimension,
                        use_bias=True,
                    )
                    if node.definition_id == "transformer.decoder_block":
                        context_dimension = context.shape[-1]
                        cross_attention_count = _attention_parameter_count(
                            query_dimension=source_dimension,
                            context_dimension=(
                                context_dimension
                                if isinstance(context_dimension, int)
                                else None
                            ),
                            heads=heads,
                            key_dimension=key_dimension,
                            value_dimension=key_dimension,
                            output_dimension=source_dimension,
                            use_bias=True,
                        )
                        attention_count = None if self_attention_count is None else cross_attention_count
                        normalization_count = 6 * source_dimension
                    else:
                        attention_count = self_attention_count
                        normalization_count = 4 * source_dimension
                    parameter_count = (
                        None
                        if attention_count is None
                        else layer_count * (attention_count + feed_forward_count + normalization_count)
                    )
                output = query
                output_ports = _enabled_output_ports(manifest, parameters)
                outputs = {port.id: output for port in output_ports}
                outputs_by_node[node_id] = outputs
                summaries.append(
                    ShapeSummary(
                        node_id=node_id,
                        outputs=outputs,
                        parameter_count=parameter_count,
                    )
                )
                continue
            if node.definition_id in {"attention.multi_head", "attention.self"}:
                layers = _optional_positive_int(parameters, "layers", 1)
                if layers != 1:
                    raise ShapeInferenceError(
                        "The supported multi-head attention node accepts layers=1"
                    )
            heads = _optional_positive_int(parameters, "num_heads", 8)
            key_dimension = _optional_positive_int(parameters, "key_dim", 64)
            value_dimension = _optional_positive_int(
                parameters,
                "value_dim",
                key_dimension,
            )
            output_dimensions, output_dimension = _attention_output_dimensions(
                parameters, query.shape[-1] if isinstance(query.shape[-1], int) else None
            )
            output = TensorSpec(
                shape=[query.shape[0], query.shape[1], *output_dimensions],
                dtype=query.dtype,
            )
            parameter_count = _attention_parameter_count(
                query_dimension=query.shape[-1] if isinstance(query.shape[-1], int) else None,
                context_dimension=context.shape[-1] if isinstance(context.shape[-1], int) else None,
                heads=heads,
                key_dimension=key_dimension,
                value_dimension=value_dimension,
                output_dimension=output_dimension,
                use_bias=bool(parameters.get("use_bias", True)),
            )
        elif rule == "math":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            if node.definition_id == "math.argmax":
                axis = _normalize_axis(_int_parameter(parameters, "axis", -1), source.rank)
                output = TensorSpec(
                    shape=[
                        dimension for index, dimension in enumerate(source.shape) if index != axis
                    ],
                    dtype=DataType.INT64,
                )
            elif node.definition_id == "math.reduce":
                axis = _normalize_axis(_int_parameter(parameters, "axis", -1), source.rank)
                if bool(parameters.get("keepdims", False)):
                    shape = list(source.shape)
                    shape[axis] = 1
                else:
                    shape = [
                        dimension for index, dimension in enumerate(source.shape) if index != axis
                    ]
                output = TensorSpec(shape=shape, dtype=source.dtype)
            else:
                if node.definition_id == "math.softmax":
                    _normalize_axis(_int_parameter(parameters, "axis", -1), source.rank)
                output = source
        elif rule == "transpose":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            axes = parameters.get("axes", [])
            if node.definition_id == "layer.permute":
                if not isinstance(axes, list):
                    raise ShapeInferenceError("Permute axes must be a list")
                permute_axes = cast(list[Any], axes)
                if len(permute_axes) != source.rank - 1:
                    raise ShapeInferenceError("Permute axes must cover every non-batch dimension")
                expected = list(range(1, source.rank))
                if any(
                    not isinstance(axis, int) or isinstance(axis, bool) for axis in permute_axes
                ):
                    raise ShapeInferenceError("Permute axes must contain integers")
                integer_axes = [int(axis) for axis in permute_axes]
                if sorted(integer_axes) != expected:
                    raise ShapeInferenceError(
                        "Permute axes must be a permutation of non-batch axes"
                    )
                output = TensorSpec(
                    shape=[source.shape[0], *[source.shape[axis] for axis in integer_axes]],
                    dtype=source.dtype,
                )
            else:
                if axes == []:
                    axes = list(reversed(range(source.rank)))
                if not isinstance(axes, list):
                    raise ShapeInferenceError("Transpose axes must be a list")
                transpose_axes = cast(list[Any], axes)
                if any(
                    not isinstance(axis, int) or isinstance(axis, bool) for axis in transpose_axes
                ):
                    raise ShapeInferenceError("Transpose axes must contain integers")
                integer_axes = [int(axis) for axis in transpose_axes]
                if sorted(integer_axes) != list(range(source.rank)):
                    raise ShapeInferenceError(
                        "Transpose axes must be a permutation of the input rank"
                    )
                output = TensorSpec(
                    shape=[source.shape[axis] for axis in integer_axes],
                    dtype=source.dtype,
                )
        elif rule == "slice":
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            source = input_specs[0]
            if node.definition_id == "math.index":
                raise ShapeInferenceError(
                    "Index shape cannot be inferred from a free-form expression"
                )
            output = TensorSpec(shape=[None] * source.rank, dtype=source.dtype)
        elif rule in {"output", "normalization", "dropout"}:
            if not input_specs:
                raise ShapeInferenceError(f"{node_id} requires an input")
            output = input_specs[0]
        else:
            if not input_specs:
                raise ShapeInferenceError(f"No input available for shape rule '{rule}'")
            output = input_specs[0]
            parameter_count = None

        output_ports = _enabled_output_ports(manifest, parameters)
        if rule == "recurrent":
            recurrent_source = input_specs[0]
            state_units = _int_parameter(parameters, "units", 64)
            recurrent_state = TensorSpec(
                shape=[recurrent_source.shape[0], state_units],
                dtype=recurrent_source.dtype,
            )
            outputs = {
                port.id: output if port.id == "output" else recurrent_state
                for port in output_ports
            }
        else:
            outputs = {port.id: output for port in output_ports}
        outputs_by_node[node_id] = outputs
        summaries.append(
            ShapeSummary(node_id=node_id, outputs=outputs, parameter_count=parameter_count)
        )

    return summaries
