from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from common.catalogs.loader import get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from models.native_runtime import build_native_model, forward_native


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
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_native_dense_values_are_deterministic_across_frameworks(
    backend: ExecutionBackend,
) -> None:
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
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_native_multi_input_multi_output_signatures_match(
    backend: ExecutionBackend,
) -> None:
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
