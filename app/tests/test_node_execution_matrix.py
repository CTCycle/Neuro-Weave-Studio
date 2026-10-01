# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from common.catalogs.loader import get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from models.native_runtime import build_native_model, forward_native
from services.validation_service import ValidationService


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
