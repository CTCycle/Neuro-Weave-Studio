# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from common.catalogs.loader import NodeCatalog, get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from domain.manifests import ParameterCapability
from models.native_runtime import build_native_model, forward_native
from services.validation_service import ValidationService
from validation.profile_compatibility import resolve_implementation_compatibility

# Consolidated from test_native_parity.py

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
    use_bias: bool = True,
) -> ModelGraph:
    parameters: dict[str, Any] = {
        "num_heads": 2,
        "key_dim": 4,
        "value_dim": 4,
        "use_bias": use_bias,
    }
    if output_shape is not None:
        parameters["output_shape"] = output_shape
    nodes = [
        node("query", "input.sequence", {"shape": [5, 8], "dtype": "float32"}),
    ]
    edges: list[EdgeInstance] = []
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
def transformer_block_graph(
    definition_id: str,
    parameters: dict[str, Any] | None = None,
) -> ModelGraph:
    configuration: dict[str, Any] = {
        "num_heads": 2,
        "key_dim": 4,
        "layers": 2
        if definition_id in {"transformer.encoder_block", "transformer.stack"}
        else 1,
        "ffn_multiplier": 2,
        "activation": "gelu",
        "dropout": 0.0,
        "pre_norm": True,
    }
    if parameters:
        configuration.update(parameters)
    if definition_id != "transformer.decoder_block":
        return chain_graph(
            "input.sequence",
            [5, 8],
            definition_id,
            configuration,
        )
    return ModelGraph(
        nodes=[
            node("query", "input.sequence", {"shape": [5, 8], "dtype": "float32"}),
            node("context", "input.sequence", {"shape": [3, 8], "dtype": "float32"}),
            node("operation", definition_id, configuration),
            node("output", "output.custom"),
        ],
        edges=[
            edge("query-operation", "query", "operation", "query"),
            edge("context-operation", "context", "operation", "context"),
            edge("operation-output", "operation", "output"),
        ],
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
            "encoder_block",
            lambda: transformer_block_graph("transformer.encoder_block"),
            (1, 5, 8),
        ),
        (
            "decoder_block",
            lambda: transformer_block_graph("transformer.decoder_block"),
            (1, 5, 8),
        ),
        (
            "transformer_stack",
            lambda: transformer_block_graph("transformer.stack"),
            (1, 5, 8),
        ),
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
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_attention_parameter_count_excludes_biases_when_disabled(
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    handle = build_native_model(
        attention_graph("attention.self", use_bias=False),
        backend,
    )

    assert handle.parameter_count() == 256

###############################################################################
@pytest.mark.parametrize("definition_id", ["transformer.encoder_block", "transformer.decoder_block"])
@pytest.mark.parametrize("pre_norm", [True, False])
@pytest.mark.parametrize("backend", native_parity_backends())
def test_native_transformer_blocks_support_pre_and_post_normalization(
    definition_id: str,
    pre_norm: bool,
    backend: ExecutionBackend,
) -> None:
    require_backend(backend)
    graph = transformer_block_graph(definition_id, {"layers": 1, "pre_norm": pre_norm})
    outputs = forward_native(build_native_model(graph, backend))

    assert len(outputs) == 1
    values = to_numpy(outputs[0])
    assert values.shape == (1, 5, 8)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()

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
            "transformer.encoder_block",
            {"num_heads": 2, "key_dim": 4, "layers": 2, "ffn_multiplier": 2},
            1200,
        ),
        (
            "transformer.decoder_block",
            {"num_heads": 2, "key_dim": 4, "layers": 1, "ffn_multiplier": 2},
            904,
        ),
        (
            "transformer.stack",
            {"num_heads": 2, "key_dim": 4, "layers": 2, "ffn_multiplier": 2},
            1200,
        ),
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
    elif definition_id in {
        "transformer.encoder_block",
        "transformer.decoder_block",
        "transformer.stack",
    }:
        graph = transformer_block_graph(definition_id, parameters)
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

# Consolidated from test_node_execution_matrix.py

###############################################################################
def _as_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(cast(list[Any] | tuple[Any, ...], value))
    return [value]

###############################################################################
def _node(
    node_id: str,
    definition_id: str,
    parameters: dict[str, Any] | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        position=Position(x=0, y=0),
        parameters=parameters or {},
    )

###############################################################################
def _edge(
    edge_id: str,
    source_node_id: str,
    source_port_id: str,
    target_node_id: str,
    target_port_id: str,
) -> EdgeInstance:
    return EdgeInstance(
        id=edge_id,
        source_node_id=source_node_id,
        source_port_id=source_port_id,
        target_node_id=target_node_id,
        target_port_id=target_port_id,
    )

###############################################################################
def _source_graph(
    definition_id: str,
    parameters: dict[str, Any],
    shape: list[int],
    dtype: str = "float32",
) -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("source", definition_id, {"shape": shape, "dtype": dtype, **parameters}),
            _node("output", "output.regression"),
        ],
        edges=[_edge("source-output", "source", "output", "output", "input")],
    )

###############################################################################
def _index_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("input", "input.tensor", {"shape": [4, 3], "dtype": "float32"}),
            _node("index", "math.index", {"expression": "x[:, 1:3]"}),
            _node("output", "output.regression"),
        ],
        edges=[
            _edge("input-index", "input", "output", "index", "input"),
            _edge("index-output", "index", "output", "output", "input"),
        ],
    )

###############################################################################
def _average_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("left", "input.tensor", {"shape": [4], "dtype": "float32"}),
            _node("middle", "input.tensor", {"shape": [4], "dtype": "float32"}),
            _node("right", "input.tensor", {"shape": [4], "dtype": "float32"}),
            _node("average", "merge.average"),
            _node("output", "output.regression"),
        ],
        edges=[
            _edge("left-average", "left", "output", "average", "inputs"),
            _edge("middle-average", "middle", "output", "average", "inputs"),
            _edge("right-average", "right", "output", "average", "inputs"),
            _edge("average-output", "average", "output", "output", "input"),
        ],
    )

###############################################################################
def _split_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("input", "input.tensor", {"shape": [4], "dtype": "float32"}),
            _node("split", "merge.split", {"sections": 2, "axis": -1}),
            _node("first", "output.regression"),
            _node("second", "output.regression"),
        ],
        edges=[
            _edge("input-split", "input", "output", "split", "input"),
            _edge("split-first", "split", "output_0", "first", "input"),
            _edge("split-second", "split", "output_1", "second", "input"),
        ],
    )

###############################################################################
def _detection_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("boxes", "input.tensor", {"shape": [5, 4], "dtype": "float32"}),
            _node("scores", "input.tensor", {"shape": [5], "dtype": "float32"}),
            _node("classes", "input.tensor", {"shape": [5], "dtype": "int32"}),
            _node("detection", "output.detection"),
        ],
        edges=[
            _edge("boxes-detection", "boxes", "output", "detection", "boxes"),
            _edge("scores-detection", "scores", "output", "detection", "scores"),
            _edge("classes-detection", "classes", "output", "detection", "classes"),
        ],
    )


###############################################################################
GRAPH_FACTORIES: list[tuple[str, Callable[[], ModelGraph], list[tuple[int, ...]]]] = [
    (
        "constant",
        lambda: _source_graph(
            "input.constant",
            {"value": [[1.0, 2.0], [3.0, 4.0]]},
            [2, 2],
        ),
        [(2, 2)],
    ),
    (
        "parameter",
        lambda: _source_graph(
            "input.parameter",
            {"initializer": "ones", "trainable": True},
            [2, 2],
        ),
        [(2, 2)],
    ),
    ("index", _index_graph, [(1, 2, 3)]),
    ("average", _average_graph, [(1, 4)]),
    ("split", _split_graph, [(1, 2), (1, 2)]),
    (
        "detection",
        _detection_graph,
        [(1, 5, 4), (1, 5), (1, 5)],
    ),
]

###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
@pytest.mark.parametrize("keras_enabled", [True, False])
@pytest.mark.parametrize(
    ("node_family", "factory", "expected_shapes"),
    GRAPH_FACTORIES,
    ids=[item[0] for item in GRAPH_FACTORIES],
)
def test_universal_nodes_validate_across_all_profiles(
    backend: ExecutionBackend,
    keras_enabled: bool,
    node_family: str,
    factory: Callable[[], ModelGraph],
    expected_shapes: list[tuple[int, ...]],
) -> None:
    del node_family, expected_shapes
    result = ValidationService(get_node_catalog()).validate(factory(), backend, keras_enabled)

    assert result.valid, result.diagnostics

###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
@pytest.mark.parametrize(
    ("node_family", "factory", "expected_shapes"),
    GRAPH_FACTORIES,
    ids=[item[0] for item in GRAPH_FACTORIES],
)
def test_universal_nodes_execute_with_native_output_ports(
    backend: ExecutionBackend,
    node_family: str,
    factory: Callable[[], ModelGraph],
    expected_shapes: list[tuple[int, ...]],
) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend],
        reason=f"{backend.value} backend is not installed",
    )
    graph = factory()
    handle = build_native_model(graph, backend, zero_parameters=True)
    outputs = forward_native(handle)

    assert len(outputs) == len(expected_shapes), node_family
    for output, expected_shape in zip(outputs, expected_shapes, strict=True):
        if backend is ExecutionBackend.PYTORCH:
            value = output.detach().cpu().numpy()
        else:
            value = np.asarray(output)
        assert value.shape == expected_shape, node_family
        assert np.isfinite(value).all(), node_family


KERAS_RUNTIME_CASES: list[tuple[str, Callable[[], ModelGraph], dict[str, np.ndarray], list[tuple[int, ...]]]] = [
    (
        "index",
        _index_graph,
        {"input": np.zeros((1, 4, 3), dtype=np.float32)},
        [(1, 2, 3)],
    ),
    (
        "average",
        _average_graph,
        {
            "left": np.zeros((1, 4), dtype=np.float32),
            "middle": np.zeros((1, 4), dtype=np.float32),
            "right": np.zeros((1, 4), dtype=np.float32),
        },
        [(1, 4)],
    ),
    (
        "split",
        _split_graph,
        {"input": np.zeros((1, 4), dtype=np.float32)},
        [(1, 2), (1, 2)],
    ),
    (
        "detection",
        _detection_graph,
        {
            "boxes": np.zeros((1, 5, 4), dtype=np.float32),
            "scores": np.zeros((1, 5), dtype=np.float32),
            "classes": np.zeros((1, 5), dtype=np.int32),
        },
        [(1, 5, 4), (1, 5), (1, 5)],
    ),
]


###############################################################################
@pytest.mark.parametrize(
    ("node_family", "factory", "inputs", "expected_shapes"),
    KERAS_RUNTIME_CASES,
    ids=[item[0] for item in KERAS_RUNTIME_CASES],
)
def test_universal_nodes_execute_through_keras_runtime_registry(
    node_family: str,
    factory: Callable[[], ModelGraph],
    inputs: dict[str, np.ndarray],
    expected_shapes: list[tuple[int, ...]],
) -> None:
    keras: Any = pytest.importorskip("keras")
    if keras.backend.backend() not in {"tensorflow", "torch", "jax"}:
        pytest.skip("The selected Keras backend is not part of the supported profile set")
    from models.keras_worker import build_model

    model: Any = cast(Any, build_model(factory()))
    outputs: Any = model(inputs, training=False)
    values = _as_values(outputs)
    assert len(values) == len(expected_shapes), node_family
    for value, expected_shape in zip(values, expected_shapes, strict=True):
        detached: Any = value.detach().cpu().numpy() if callable(getattr(value, "detach", None)) else np.asarray(value)
        assert detached.shape == expected_shape, node_family
        assert np.isfinite(detached).all(), node_family


###############################################################################
@pytest.mark.parametrize(
    ("node_family", "factory", "trainable_count"),
    [
        (
            "constant",
            lambda: _source_graph(
                "input.constant",
                {"value": [[1.0, 2.0], [3.0, 4.0]]},
                [2, 2],
            ),
            0,
        ),
        (
            "parameter",
            lambda: _source_graph(
                "input.parameter",
                {"initializer": "ones", "trainable": True},
                [2, 2],
            ),
            1,
        ),
    ],
    ids=["constant", "parameter"],
)
def test_keras_static_sources_have_no_public_inputs_and_reload(
    node_family: str,
    factory: Callable[[], ModelGraph],
    trainable_count: int,
    tmp_path: Path,
) -> None:
    keras: Any = pytest.importorskip("keras")
    if keras.backend.backend() not in {"tensorflow", "torch", "jax"}:
        pytest.skip("The selected Keras backend is not part of the supported profile set")
    from models.keras_worker import build_model

    model: Any = cast(Any, build_model(factory()))
    assert not getattr(model, "inputs", ())
    assert len(model.trainable_variables) == trainable_count, node_family
    initial: Any = model({}, training=False)
    initial_values = _as_values(initial)
    assert initial_values[0].shape == (2, 2), node_family

    path = tmp_path / f"{node_family}.keras"
    model.save(path)
    restored: Any = keras.models.load_model(path)
    assert not getattr(restored, "inputs", ())
    assert len(restored.trainable_variables) == trainable_count, node_family
    output: Any = restored({}, training=False)
    values = _as_values(output)
    assert values[0].shape == (2, 2), node_family

# Consolidated from test_backend_compatibility_consolidation.py

###############################################################################
PROFILES = [
    (ExecutionBackend.TENSORFLOW, True),
    (ExecutionBackend.PYTORCH, True),
    (ExecutionBackend.JAX, True),
    (ExecutionBackend.TENSORFLOW, False),
    (ExecutionBackend.PYTORCH, False),
    (ExecutionBackend.JAX, False),
]

###############################################################################
def make_node(
    node_id: str,
    definition_id: str,
    parameters: dict[str, object] | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        position=Position(x=0, y=0),
        parameters=parameters or {},
    )

###############################################################################
def _test_backend_compatibility_consolidation_chain_graph(
    definition_id: str,
    parameters: dict[str, object] | None = None,
    *,
    input_definition: str = "input.tabular",
    input_parameters: dict[str, object] | None = None,
    output_definition: str = "output.regression",
) -> ModelGraph:
    return ModelGraph(
        nodes=[
            make_node("input", input_definition, input_parameters or {"shape": [4], "dtype": "float32"}),
            make_node("operation", definition_id, parameters),
            make_node("output", output_definition),
        ],
        edges=[
            EdgeInstance(
                id="input-operation",
                source_node_id="input",
                source_port_id="output",
                target_node_id="operation",
                target_port_id="input",
            ),
            EdgeInstance(
                id="operation-output",
                source_node_id="operation",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )


###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_common_dense_chain_is_valid_for_all_six_static_profiles(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        _test_backend_compatibility_consolidation_chain_graph(
            "layer.dense",
            {"units": 3, "activation": "softmax"},
            output_definition="output.classification",
        ),
        execution_backend,
        keras_enabled,
    )

    assert result.valid, result.diagnostics
    assert not any(
        item.code in {"unsupported_implementation", "unverified_implementation"}
        for item in result.diagnostics
    )

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_simple_rnn_is_actionable_in_every_profile(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        _test_backend_compatibility_consolidation_chain_graph(
            "recurrent.simple_rnn",
            {"units": 4},
            input_definition="input.sequence",
            input_parameters={"shape": [12, 8], "dtype": "float32"},
            output_definition="output.sequence",
        ),
        execution_backend,
        keras_enabled,
    )

    assert result.valid, result.diagnostics

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_dense_initializer_accepts_keras_but_rejects_non_default_native_values(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        _test_backend_compatibility_consolidation_chain_graph(
            "layer.dense",
            {"units": 2, "kernel_initializer": "he_normal"},
        ),
        execution_backend,
        keras_enabled,
    )

    if keras_enabled:
        assert result.valid, result.diagnostics
    else:
        assert not result.valid
        assert any(
            item.code == "unsupported_parameter_value"
            and item.parameter_id == "kernel_initializer"
            for item in result.diagnostics
        )

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_native_initializer_defaults_remain_valid_while_non_defaults_are_blocked(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        _test_backend_compatibility_consolidation_chain_graph(
            "layer.dense",
            {"units": 2, "kernel_initializer": "glorot_uniform"},
        ),
        execution_backend,
        keras_enabled,
    )

    assert result.valid, result.diagnostics

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_lstm_unit_forget_bias_uses_the_selected_backend_mapping(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        _test_backend_compatibility_consolidation_chain_graph(
            "recurrent.lstm",
            {"units": 4, "unit_forget_bias": False},
            input_definition="input.sequence",
            input_parameters={"shape": [12, 8], "dtype": "float32"},
            output_definition="output.sequence",
        ),
        execution_backend,
        keras_enabled,
    )

    expected_valid = keras_enabled or execution_backend is ExecutionBackend.PYTORCH
    assert result.valid is expected_valid, result.diagnostics
    if not expected_valid:
        assert any(
            item.code == "unsupported_parameter_option"
            and item.parameter_id == "unit_forget_bias"
            for item in result.diagnostics
        )

###############################################################################
def test_keras_backend_exclusion_has_a_distinct_reason_code() -> None:
    base_catalog = get_node_catalog()
    base_manifest = base_catalog.get("layer.dense")
    excluded_mapping = base_manifest.implementations["keras"].model_copy(
        update={
            "execution_backends": {
                ExecutionBackend.TENSORFLOW: False,
                ExecutionBackend.PYTORCH: True,
                ExecutionBackend.JAX: True,
            }
        }
    )
    implementations = dict(base_manifest.implementations)
    implementations["keras"] = excluded_mapping
    excluded_manifest = base_manifest.model_copy(
        deep=True,
        update={
            "id": "test.keras_excluded_dense",
            "display_name": "Keras excluded Dense",
            "implementations": implementations,
        },
    )
    catalog = NodeCatalog([*base_catalog.manifests, excluded_manifest])

    result = ValidationService(catalog).validate(
        _test_backend_compatibility_consolidation_chain_graph("test.keras_excluded_dense"),
        ExecutionBackend.TENSORFLOW,
        True,
    )

    diagnostic = next(item for item in result.diagnostics if item.node_id == "operation")
    assert not result.valid
    assert diagnostic.details["reason_code"] == "backend_excluded"
    assert "Keras" in diagnostic.message

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_synthetic_unverified_node_mapping_is_unavailable(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    base_catalog = get_node_catalog()
    base_manifest = base_catalog.get("layer.dense")
    implementation_key = "keras" if keras_enabled else execution_backend.value
    implementations = dict(base_manifest.implementations)
    implementations[implementation_key] = implementations[implementation_key].model_copy(
        update={"status": "unverified"}
    )
    unverified_manifest = base_manifest.model_copy(
        deep=True,
        update={
            "id": "test.unverified_dense",
            "display_name": "Unverified Dense",
            "implementations": implementations,
        },
    )
    catalog = NodeCatalog([*base_catalog.manifests, unverified_manifest])

    result = ValidationService(catalog).validate(
        _test_backend_compatibility_consolidation_chain_graph("test.unverified_dense"),
        execution_backend,
        keras_enabled,
    )

    diagnostic = next(item for item in result.diagnostics if item.node_id == "operation")
    assert not result.valid
    assert diagnostic.code == "unverified_implementation"
    assert diagnostic.details["reason_code"] == "unverified_implementation"

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_synthetic_unverified_parameter_mapping_is_unavailable(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    base_catalog = get_node_catalog()
    base_manifest = base_catalog.get("layer.dense")
    implementation_key = "keras" if keras_enabled else execution_backend.value
    native_mapping = base_manifest.implementations[implementation_key]
    parameters = dict(native_mapping.parameters)
    parameters["units"] = ParameterCapability(supported=True, status="unverified")
    implementations = dict(base_manifest.implementations)
    implementations[implementation_key] = native_mapping.model_copy(
        update={"parameters": parameters}
    )
    unverified_manifest = base_manifest.model_copy(
        deep=True,
        update={
            "id": "test.unverified_dense_parameter",
            "display_name": "Unverified Dense Parameter",
            "implementations": implementations,
        },
    )
    catalog = NodeCatalog([*base_catalog.manifests, unverified_manifest])

    result = ValidationService(catalog).validate(
        _test_backend_compatibility_consolidation_chain_graph("test.unverified_dense_parameter", {"units": 2}),
        execution_backend,
        keras_enabled,
    )

    assert not result.valid
    assert any(
        item.code == "unverified_parameter_mapping"
        and item.parameter_id == "units"
        for item in result.diagnostics
    )

###############################################################################
def test_profile_resolver_distinguishes_missing_unsupported_and_unverified_mappings() -> None:
    base = get_node_catalog().get("layer.dense")
    missing = base.model_copy(deep=True, update={"implementations": {}})
    unsupported_mapping = base.implementations["pytorch"].model_copy(
        update={"supported": False}
    )
    unsupported = base.model_copy(
        deep=True,
        update={"implementations": {"pytorch": unsupported_mapping}},
    )
    unverified_mapping = base.implementations["pytorch"].model_copy(
        update={"status": "unverified"}
    )
    unverified = base.model_copy(
        deep=True,
        update={"implementations": {"pytorch": unverified_mapping}},
    )

    assert resolve_implementation_compatibility(missing, ExecutionBackend.PYTORCH, False).reason_code == "missing_implementation"
    assert resolve_implementation_compatibility(unsupported, ExecutionBackend.PYTORCH, False).reason_code == "unsupported_implementation"
    assert resolve_implementation_compatibility(unverified, ExecutionBackend.PYTORCH, False).reason_code == "unverified_implementation"
