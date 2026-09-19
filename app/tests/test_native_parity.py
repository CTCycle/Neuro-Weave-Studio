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
        ("cnn", cnn_graph, (1, 4, 4, 2)),
        ("recurrent", recurrent_graph, (1, 3)),
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
