from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from common.catalogs.loader import get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from models.native_runtime import build_native_model, forward_native

REQUIRED_NATIVE_BACKEND_ENV = "NEURO_WEAVE_REQUIRED_NATIVE_BACKEND"


###############################################################################
def configured_native_backend() -> ExecutionBackend | None:
    configured = os.getenv(REQUIRED_NATIVE_BACKEND_ENV, "").strip().lower()
    if not configured:
        return None
    try:
        return ExecutionBackend(configured)
    except ValueError as exc:
        valid_backends = ", ".join(backend.value for backend in ExecutionBackend)
        raise ValueError(
            f"{REQUIRED_NATIVE_BACKEND_ENV} must be one of {valid_backends}; received {configured!r}"
        ) from exc


###############################################################################
def native_parity_backends() -> tuple[ExecutionBackend, ...]:
    """Select the CI matrix backend or all optional runtimes for local runs."""

    configured = configured_native_backend()
    return (configured,) if configured is not None else tuple(ExecutionBackend)


###############################################################################
def require_backend(backend: ExecutionBackend) -> None:
    module_name = {
        ExecutionBackend.TENSORFLOW: "tensorflow",
        ExecutionBackend.PYTORCH: "torch",
        ExecutionBackend.JAX: "jax",
    }[backend]
    if configured_native_backend() is backend:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            pytest.fail(
                f"Required {backend.value} runtime could not import {module_name!r}: {exc}",
                pytrace=False,
            )
        return
    pytest.importorskip(module_name, reason=f"{backend.value} backend is not installed")

###############################################################################
def node(
    node_id: str, definition_id: str, parameters: dict[str, Any] | None = None
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        definition_version=1,
        position=Position(x=0, y=0),
        parameters=parameters or {},
    )

###############################################################################
def edge(edge_id: str, source: str, target: str, target_port: str = "input") -> EdgeInstance:
    return EdgeInstance(
        id=edge_id,
        source_node_id=source,
        source_port_id="output",
        target_node_id=target,
        target_port_id=target_port,
    )

###############################################################################
def dense_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            node("dense", "layer.dense", {"units": 3, "activation": "softmax"}),
            node("output", "output.classification"),
        ],
        edges=[edge("e1", "input", "dense"), edge("e2", "dense", "output")],
    )

###############################################################################
def multi_io_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            node("left", "input.tabular", {"shape": [2], "dtype": "float32"}),
            node("right", "input.tabular", {"shape": [3], "dtype": "float32"}),
            node("join", "merge.concatenate", {"axis": -1}),
            node("first", "output.regression"),
            node("second", "output.classification"),
        ],
        edges=[
            edge("e1", "left", "join", "inputs"),
            edge("e2", "right", "join", "inputs"),
            edge("e3", "join", "first"),
            edge("e4", "join", "second"),
        ],
    )


###############################################################################
def chain_graph(
    input_definition: str,
    input_shape: list[int],
    definition_id: str,
    parameters: dict[str, Any] | None = None,
) -> ModelGraph:
    return ModelGraph(
        nodes=[
            node("input", input_definition, {"shape": input_shape, "dtype": "float32"}),
            node("operation", definition_id, parameters),
            node("output", "output.custom"),
        ],
        edges=[
            edge("input-operation", "input", "operation"),
            edge("operation-output", "operation", "output"),
        ],
    )


###############################################################################
def activation_graph() -> ModelGraph:
    return chain_graph(
        "input.tensor",
        [4],
        "layer.activation",
        {"activation": "relu"},
    )


###############################################################################
def layer_reshape_graph() -> ModelGraph:
    return chain_graph(
        "input.tensor",
        [2, 2],
        "layer.reshape",
        {"target_shape": [4]},
    )


###############################################################################
def layer_permute_graph() -> ModelGraph:
    return chain_graph(
        "input.tensor",
        [2, 3],
        "layer.permute",
        {"axes": [2, 1]},
    )


###############################################################################
def layer_repeat_graph() -> ModelGraph:
    return chain_graph(
        "input.tensor",
        [4],
        "layer.repeat",
        {"repeats": 2, "axis": 1},
    )


###############################################################################
def normalization_graph(definition_id: str, parameters: dict[str, Any] | None = None) -> ModelGraph:
    return chain_graph(
        "input.tensor",
        [4],
        definition_id,
        parameters,
    )


###############################################################################
def spatial_graph(
    definition_id: str,
    input_shape: list[int],
    parameters: dict[str, Any],
) -> ModelGraph:
    return chain_graph("input.tensor", input_shape, definition_id, parameters)


###############################################################################
def cnn_graph() -> ModelGraph:
    return chain_graph(
        "input.image",
        [8, 8, 1],
        "conv.conv2d",
        {
            "filters": 2,
            "kernel_size": [3, 3],
            "strides": [2, 2],
            "padding": "same",
            "activation": "relu",
            "data_format": "channels_last",
            "dilation_rate": [1, 1],
            "groups": 1,
            "use_bias": True,
        },
    )


###############################################################################
def recurrent_graph() -> ModelGraph:
    return chain_graph(
        "input.sequence",
        [6, 4],
        "recurrent.lstm",
        {"units": 3, "return_sequences": False},
    )


###############################################################################
def recurrent_wrapper_graph(
    definition_id: str,
    parameters: dict[str, Any],
) -> ModelGraph:
    return chain_graph("input.sequence", [6, 4], definition_id, parameters)


###############################################################################
def simple_rnn_state_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [6, 4], "dtype": "float32"}),
            node(
                "operation",
                "recurrent.simple_rnn",
                {"units": 3, "return_sequences": True, "return_state": True},
            ),
        ],
        edges=[edge("input-operation", "input", "operation")],
    )


###############################################################################
def positional_encoding_graph(mode: str = "sinusoidal") -> ModelGraph:
    return chain_graph(
        "input.sequence",
        [6, 8],
        "transformer.positional_encoding",
        {"mode": mode},
    )


###############################################################################
def attention_mask_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            node("input", "input.mask", {"shape": [6, 6], "dtype": "bool"}),
            node("operation", "transformer.attention_mask"),
            node("output", "output.custom"),
        ],
        edges=[edge("input-operation", "input", "operation"), edge("operation-output", "operation", "output")],
    )


###############################################################################
def transformer_residual_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            node("left", "input.tensor", {"shape": [6, 8], "dtype": "float32"}),
            node("right", "input.tensor", {"shape": [6, 8], "dtype": "float32"}),
            node("residual", "transformer.residual"),
            node("output", "output.custom"),
        ],
        edges=[
            edge("left-residual", "left", "residual", "inputs"),
            edge("right-residual", "right", "residual", "inputs"),
            edge("residual-output", "residual", "output"),
        ],
    )


###############################################################################
def attention_graph(
    definition_id: str,
    *,
    context: bool = False,
    output_shape: list[int] | None = None,
) -> ModelGraph:
    parameters: dict[str, Any] = {
        "num_heads": 2,
        "key_dim": 4,
        "value_dim": 4,
        "use_bias": True,
    }
    if output_shape is not None:
        parameters["output_shape"] = output_shape
    nodes = [
        node("query", "input.sequence", {"shape": [5, 8], "dtype": "float32"}),
    ]
    edges = []
    if context:
        nodes.append(node("context", "input.sequence", {"shape": [3, 8], "dtype": "float32"}))
    nodes.extend(
        [
            node("attention", definition_id, parameters),
            node("output", "output.custom"),
        ]
    )
    edges.append(edge("query-attention", "query", "attention", "query"))
    if context:
        edges.append(edge("context-attention", "context", "attention", "context"))
    edges.append(edge("attention-output", "attention", "output"))
    return ModelGraph(nodes=nodes, edges=edges)


###############################################################################
def feed_forward_graph(parameters: dict[str, Any] | None = None) -> ModelGraph:
    configuration: dict[str, Any] = {
        "activation": "gelu",
        "ffn_multiplier": 2,
        "dropout": 0.0,
    }
    if parameters:
        configuration.update(parameters)
    return chain_graph(
        "input.sequence",
        [5, 8],
        "transformer.feed_forward",
        configuration,
    )


###############################################################################
def reshape_flatten_graph() -> ModelGraph:
    graph = chain_graph(
        "input.tensor",
        [2, 2],
        "math.reshape",
        {"target_shape": [4]},
    )
    graph.nodes.insert(2, node("flatten", "layer.flatten"))
    graph.edges = [
        edge("input-reshape", "input", "operation"),
        edge("reshape-flatten", "operation", "flatten"),
        edge("flatten-output", "flatten", "output"),
    ]
    return graph


###############################################################################
def merge_add_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            node("left", "input.tensor", {"shape": [4], "dtype": "float32"}),
            node("right", "input.tensor", {"shape": [4], "dtype": "float32"}),
            node("add", "merge.add"),
            node("output", "output.custom"),
        ],
        edges=[
            edge("left-add", "left", "add", "inputs"),
            edge("right-add", "right", "add", "inputs"),
            edge("add-output", "add", "output"),
        ],
    )


###############################################################################
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_dense_values_are_deterministic_across_frameworks(
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    handle = build_native_model(dense_graph(), backend)
    outputs = forward_native(handle)

    assert len(outputs) == 1
    values = to_numpy(outputs[0])
    assert values.shape == (1, 3)
    assert values.dtype == np.float32
    np.testing.assert_allclose(
        values, np.full((1, 3), 1 / 3, dtype=np.float32), rtol=1e-5, atol=1e-6
    )


###############################################################################
@pytest.mark.parametrize(
    ("case_name", "graph_factory", "expected_shape"),
    [
        ("activation", activation_graph, (1, 4)),
        ("layer_reshape", layer_reshape_graph, (1, 4)),
        ("layer_permute", layer_permute_graph, (1, 3, 2)),
        ("layer_repeat", layer_repeat_graph, (1, 2, 4)),
        ("group_norm", lambda: normalization_graph("layer.group_norm", {"groups": 2}), (1, 4)),
        ("prelu", lambda: normalization_graph("layer.prelu"), (1, 4)),
        ("rms_norm", lambda: normalization_graph("layer.rms_norm"), (1, 4)),
        ("unit_norm", lambda: normalization_graph("layer.unit_norm"), (1, 4)),
        (
            "max_pool1d",
            lambda: spatial_graph(
                "conv.max_pool1d", [8, 2], {"pool_size": [2], "strides": [2]}
            ),
            (1, 4, 2),
        ),
        (
            "average_pool1d",
            lambda: spatial_graph(
                "conv.average_pool1d", [8, 2], {"pool_size": [2], "strides": [2]}
            ),
            (1, 4, 2),
        ),
        (
            "max_pool2d_channels_first",
            lambda: spatial_graph(
                "conv.max_pool2d",
                [2, 8, 8],
                {"pool_size": [2, 2], "strides": [2, 2], "data_format": "channels_first"},
            ),
            (1, 2, 4, 4),
        ),
        (
            "average_pool2d",
            lambda: spatial_graph(
                "conv.average_pool2d", [8, 8, 2], {"pool_size": [2, 2], "strides": [2, 2]}
            ),
            (1, 4, 4, 2),
        ),
        (
            "max_pool3d",
            lambda: spatial_graph(
                "conv.max_pool3d", [4, 4, 4, 2], {"pool_size": [2, 2, 2], "strides": [2, 2, 2]}
            ),
            (1, 2, 2, 2, 2),
        ),
        (
            "average_pool3d",
            lambda: spatial_graph(
                "conv.average_pool3d", [4, 4, 4, 2], {"pool_size": [2, 2, 2], "strides": [2, 2, 2]}
            ),
            (1, 2, 2, 2, 2),
        ),
        (
            "global_pool1d",
            lambda: spatial_graph("conv.global_pool1d", [8, 2], {"mode": "maximum"}),
            (1, 2),
        ),
        (
            "global_pool2d_keepdims",
            lambda: spatial_graph(
                "conv.global_pool2d", [4, 4, 2], {"mode": "maximum", "keepdims": True}
            ),
            (1, 1, 1, 2),
        ),
        (
            "global_pool3d",
            lambda: spatial_graph("conv.global_pool3d", [3, 3, 3, 2], {"mode": "average"}),
            (1, 2),
        ),
        (
            "upsampling",
            lambda: spatial_graph(
                "conv.upsampling", [2, 3, 1], {"size": [2, 3], "interpolation": "bilinear"}
            ),
            (1, 4, 9, 1),
        ),
        (
            "padding",
            lambda: spatial_graph(
                "conv.padding", [4, 5, 1], {"padding": [[1, 2], [2, 1]]}
            ),
            (1, 7, 8, 1),
        ),
        (
            "cropping",
            lambda: spatial_graph(
                "conv.cropping", [7, 8, 1], {"cropping": [[1, 2], [2, 1]]}
            ),
            (1, 4, 5, 1),
        ),
        (
            "conv3d",
            lambda: spatial_graph(
                "conv.conv3d",
                [4, 4, 4, 1],
                {
                    "filters": 2,
                    "kernel_size": [3, 3, 3],
                    "strides": [1, 1, 1],
                    "padding": "same",
                    "data_format": "channels_last",
                    "dilation_rate": [1, 1, 1],
                    "groups": 1,
                    "use_bias": True,
                },
            ),
            (1, 4, 4, 4, 2),
        ),
        (
            "depthwise",
            lambda: spatial_graph(
                "conv.depthwise",
                [4, 4, 2],
                {
                    "dimensions": 2,
                    "depth_multiplier": 2,
                    "kernel_size": [3, 3],
                    "strides": [1, 1],
                    "padding": "same",
                    "data_format": "channels_last",
                    "dilation_rate": [1, 1],
                    "use_bias": True,
                },
            ),
            (1, 4, 4, 4),
        ),
        (
            "separable",
            lambda: spatial_graph(
                "conv.separable",
                [4, 4, 2],
                {
                    "dimensions": 2,
                    "filters": 3,
                    "depth_multiplier": 2,
                    "kernel_size": [3, 3],
                    "strides": [1, 1],
                    "padding": "same",
                    "data_format": "channels_last",
                    "dilation_rate": [1, 1],
                    "use_bias": True,
                },
            ),
            (1, 4, 4, 3),
        ),
        (
            "transposed",
            lambda: spatial_graph(
                "conv.transposed",
                [4, 4, 1],
                {
                    "dimensions": 2,
                    "filters": 2,
                    "kernel_size": [3, 3],
                    "strides": [2, 2],
                    "padding": "valid",
                    "output_padding": [1, 1],
                    "data_format": "channels_last",
                    "dilation_rate": [1, 1],
                    "use_bias": True,
                },
        ),
        (1, 10, 10, 2),
        ),
        ("cnn", cnn_graph, (1, 4, 4, 2)),
        ("recurrent", recurrent_graph, (1, 3)),
        ("positional_encoding", positional_encoding_graph, (1, 6, 8)),
        ("transformer_residual", transformer_residual_graph, (1, 6, 8)),
        ("feed_forward", feed_forward_graph, (1, 5, 8)),
        (
            "simple_rnn",
            lambda: recurrent_wrapper_graph(
                "recurrent.simple_rnn", {"units": 3, "return_sequences": False}
            ),
            (1, 3),
        ),
        (
            "bidirectional",
            lambda: recurrent_wrapper_graph(
                "recurrent.bidirectional",
                {"cell": "simple_rnn", "units": 3, "merge_mode": "concat"},
            ),
            (1, 6),
        ),
        (
            "stack",
            lambda: recurrent_wrapper_graph(
                "recurrent.stack",
                {"cell": "simple_rnn", "units": 3, "layers": 2},
            ),
            (1, 3),
        ),
        ("reshape_flatten", reshape_flatten_graph, (1, 4)),
        ("merge_add", merge_add_graph, (1, 4)),
    ],
)
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_representative_cases(
    case_name: str,
    graph_factory: Callable[[], ModelGraph],
    expected_shape: tuple[int, ...],
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    graph = graph_factory()
    catalog = get_node_catalog()
    for graph_node in graph.nodes:
        mapping = catalog.get(graph_node.definition_id).implementations.get(backend.value)
        assert mapping is not None and mapping.supported, (
            f"{case_name} uses unsupported native {backend.value} node "
            f"{graph_node.definition_id}"
        )

    outputs = forward_native(build_native_model(graph, backend))

    assert len(outputs) == 1
    value = to_numpy(outputs[0])
    assert value.shape == expected_shape
    assert value.dtype == np.float32


###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "parameters", "expected_shape"),
    [
        (
            "recurrent.bidirectional",
            {"cell": "simple_rnn", "units": 3, "merge_mode": "sum"},
            (1, 3),
        ),
        (
            "recurrent.bidirectional",
            {"cell": "simple_rnn", "units": 3, "merge_mode": "mul"},
            (1, 3),
        ),
        (
            "recurrent.bidirectional",
            {"cell": "simple_rnn", "units": 3, "merge_mode": "ave"},
            (1, 3),
        ),
        (
            "recurrent.bidirectional",
            {"cell": "simple_rnn", "units": 3, "merge_mode": "concat", "return_sequences": True},
            (1, 6, 6),
        ),
    ],
)
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_bidirectional_merge_modes(
    definition_id: str,
    parameters: dict[str, Any],
    expected_shape: tuple[int, ...],
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    outputs = forward_native(
        build_native_model(recurrent_wrapper_graph(definition_id, parameters), backend)
    )

    assert len(outputs) == 1
    assert to_numpy(outputs[0]).shape == expected_shape


###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "cell"),
    [
        ("recurrent.bidirectional", "simple_rnn"),
        ("recurrent.bidirectional", "lstm"),
        ("recurrent.bidirectional", "gru"),
        ("recurrent.stack", "simple_rnn"),
        ("recurrent.stack", "lstm"),
        ("recurrent.stack", "gru"),
    ],
)
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_recurrent_wrapper_cells(
    definition_id: str,
    cell: str,
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    parameters: dict[str, Any] = {"cell": cell, "units": 3}
    if definition_id == "recurrent.stack":
        parameters["layers"] = 2
    outputs = forward_native(
        build_native_model(recurrent_wrapper_graph(definition_id, parameters), backend)
    )

    assert len(outputs) == 1
    expected_width = 6 if definition_id == "recurrent.bidirectional" else 3
    assert to_numpy(outputs[0]).shape == (1, expected_width)


###############################################################################
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_simple_rnn_state_port_matches_sequence_output(
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    outputs = forward_native(build_native_model(simple_rnn_state_graph(), backend))

    assert len(outputs) == 2
    sequence = to_numpy(outputs[0])
    hidden_state = to_numpy(outputs[1])
    assert sequence.shape == (1, 6, 3)
    assert hidden_state.shape == (1, 3)
    np.testing.assert_allclose(hidden_state, sequence[:, -1, :])


###############################################################################
@pytest.mark.parametrize("mode", ["sinusoidal", "learned"])
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_positional_encoding_is_finite_and_shape_stable(
    mode: str,
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    outputs = forward_native(build_native_model(positional_encoding_graph(mode), backend))

    assert len(outputs) == 1
    values = to_numpy(outputs[0])
    assert values.shape == (1, 6, 8)
    assert np.isfinite(values).all()


###############################################################################
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_attention_mask_preserves_boolean_tensor(
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    outputs = forward_native(build_native_model(attention_mask_graph(), backend))

    assert len(outputs) == 1
    values = to_numpy(outputs[0])
    assert values.shape == (1, 6, 6)
    assert values.dtype == np.bool_


###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "context", "output_shape", "expected_shape"),
    [
        ("attention.self", False, None, (1, 5, 8)),
        ("attention.cross", True, None, (1, 5, 8)),
        ("attention.multi_head", False, [2, 4], (1, 5, 2, 4)),
    ],
)
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_attention_primitives_emit_finite_port_aligned_outputs(
    definition_id: str,
    context: bool,
    output_shape: list[int] | None,
    expected_shape: tuple[int, ...],
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    outputs = forward_native(
        build_native_model(
            attention_graph(definition_id, context=context, output_shape=output_shape),
            backend,
        )
    )

    assert len(outputs) == 1
    values = to_numpy(outputs[0])
    assert values.shape == expected_shape
    assert values.dtype == np.float32
    assert np.isfinite(values).all()


###############################################################################
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_attention_parameter_count_includes_all_projections(
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    handle = build_native_model(attention_graph("attention.self"), backend)

    assert handle.parameter_count() == 288


###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "parameters", "expected_parameter_count"),
    [
        ("layer.group_norm", {"groups": 2}, 8),
        ("layer.prelu", {}, 4),
        ("layer.rms_norm", {}, 4),
        ("layer.unit_norm", {}, 0),
        (
            "recurrent.simple_rnn",
            {"units": 3},
            {"tensorflow": 24, "pytorch": 27, "jax": 24},
        ),
        (
            "recurrent.bidirectional",
            {"cell": "simple_rnn", "units": 3, "merge_mode": "concat"},
            {"tensorflow": 48, "pytorch": 54, "jax": 48},
        ),
        (
            "recurrent.stack",
            {"cell": "simple_rnn", "units": 3, "layers": 2},
            {"tensorflow": 45, "pytorch": 51, "jax": 45},
        ),
        (
            "recurrent.stack",
            {"cell": "lstm", "units": 3, "layers": 2},
            {"tensorflow": 180, "pytorch": 204, "jax": 180},
        ),
        (
            "transformer.positional_encoding",
            {"mode": "learned"},
            {"tensorflow": 48, "pytorch": 48, "jax": 48},
        ),
        ("transformer.feed_forward", {"ffn_multiplier": 2}, 280),
        (
            "conv.conv3d",
            {
                "filters": 2,
                "kernel_size": [3, 3, 3],
                "strides": [1, 1, 1],
                "padding": "same",
                "data_format": "channels_last",
                "dilation_rate": [1, 1, 1],
                "groups": 1,
                "use_bias": True,
            },
            56,
        ),
        (
            "conv.depthwise",
            {
                "dimensions": 2,
                "depth_multiplier": 2,
                "kernel_size": [3, 3],
                "strides": [1, 1],
                "padding": "same",
                "data_format": "channels_last",
                "dilation_rate": [1, 1],
                "use_bias": True,
            },
            40,
        ),
        (
            "conv.separable",
            {
                "dimensions": 2,
                "filters": 3,
                "depth_multiplier": 2,
                "kernel_size": [3, 3],
                "strides": [1, 1],
                "padding": "same",
                "data_format": "channels_last",
                "dilation_rate": [1, 1],
                "use_bias": True,
            },
            51,
        ),
        (
            "conv.transposed",
            {
                "dimensions": 2,
                "filters": 2,
                "kernel_size": [3, 3],
                "strides": [2, 2],
                "padding": "valid",
                "output_padding": [0, 0],
                "data_format": "channels_last",
                "dilation_rate": [1, 1],
                "use_bias": True,
            },
            20,
        ),
    ],
)
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_parameter_counts(
    definition_id: str,
    parameters: dict[str, Any],
    expected_parameter_count: int | dict[str, int],
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    if definition_id in {"conv.depthwise", "conv.separable"}:
        input_shape = [4, 4, 2]
    elif definition_id == "conv.conv3d":
        input_shape = [4, 4, 4, 1]
    else:
        input_shape = [4, 4, 1]
    if definition_id.startswith("conv."):
        graph = spatial_graph(definition_id, input_shape, parameters)
    elif definition_id.startswith("recurrent."):
        graph = recurrent_wrapper_graph(definition_id, parameters)
    elif definition_id == "transformer.positional_encoding":
        graph = positional_encoding_graph(str(parameters.get("mode", "sinusoidal")))
    elif definition_id == "transformer.feed_forward":
        graph = feed_forward_graph(parameters)
    else:
        graph = normalization_graph(definition_id, parameters)
    handle = build_native_model(graph, backend)

    expected = (
        expected_parameter_count
        if isinstance(expected_parameter_count, int)
        else expected_parameter_count[backend.value]
    )
    assert handle.parameter_count() == expected

###############################################################################
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_multi_input_multi_output_signatures_match(
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    graph = multi_io_graph()
    result = build_native_model(graph, backend)
    outputs = forward_native(result)

    assert len(result.inputs) == 2
    assert len(outputs) == 2
    assert [tuple(to_numpy(value).shape) for value in outputs] == [(1, 5), (1, 5)]
    np.testing.assert_allclose(to_numpy(outputs[0]), np.zeros((1, 5), dtype=np.float32))
    np.testing.assert_allclose(to_numpy(outputs[1]), np.zeros((1, 5), dtype=np.float32))

###############################################################################
def test_native_catalog_is_capability_driven() -> None:
    catalog = get_node_catalog()

    for manifest in catalog.manifests:
        for backend in ("tensorflow", "pytorch", "jax"):
            mapping = manifest.implementations.get(backend)
            if mapping is None or not mapping.supported:
                continue
            assert mapping.symbol or mapping.adapter, (
                f"{manifest.id} has no declarative {backend} implementation"
            )

    assert catalog.get("conv.conv1d").implementations["tensorflow"].supported
    assert catalog.get("conv.conv1d").implementations["pytorch"].supported
    assert catalog.get("conv.conv1d").implementations["jax"].supported
    assert "attention.multi_head" not in {
        manifest.id for manifest in catalog.manifests if manifest.operation is not None
    }
    assert "transformer.encoder_block" not in {
        manifest.id for manifest in catalog.manifests if manifest.operation is not None
    }

###############################################################################
def to_numpy(value: Any) -> np.ndarray:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    on_cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    return np.asarray(on_cpu)
