from __future__ import annotations

from typing import Any

import pytest

from common.catalogs.loader import get_node_catalog
from compilation.compiler import compile_to_ir
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from exporters.pytorch_exporter import PyTorchSourceExporter
from services.validation_service import ValidationService

###############################################################################
def node(
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
def edge(edge_id: str, source: str, target: str, target_port: str = "input") -> EdgeInstance:
    return EdgeInstance(
        id=edge_id,
        source_node_id=source,
        source_port_id="output",
        target_node_id=target,
        target_port_id=target_port,
    )

###############################################################################
def build_model(graph: ModelGraph) -> Any:
    torch = pytest.importorskip("torch")
    validation = ValidationService(get_node_catalog()).validate(graph, ExecutionBackend.PYTORCH, False)
    assert validation.valid, validation.diagnostics
    ir = compile_to_ir(
        graph,
        ExecutionBackend.PYTORCH,
        False,
        get_node_catalog(),
        shape_summaries=validation.shape_summaries,
    )
    namespace: dict[str, Any] = {"__name__": "generated_pytorch_model"}
    exec(PyTorchSourceExporter().generate(graph, ir), namespace)
    model = namespace["GeneratedModel"]()
    model.eval()
    return torch, model

###############################################################################
def test_generated_pytorch_dense_and_dropout_execute() -> None:
    graph = ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("dense", "layer.dense", {"units": 3, "activation": "relu", "use_bias": False}),
            node("dropout", "layer.dropout", {"rate": 0.0}),
            node("output", "output.regression"),
        ],
        edges=[
            edge("e1", "input", "dense"),
            edge("e2", "dense", "dropout"),
            edge("e3", "dropout", "output"),
        ],
    )
    torch, model = build_model(graph)
    result = model(torch.randn(2, 4))
    assert tuple(result.shape) == (2, 3)

###############################################################################
def test_generated_pytorch_conv_pool_preserves_canonical_channels_last_shape() -> None:
    graph = ModelGraph(
        nodes=[
            node("image", "input.image", {"shape": [None, 7, 9, 3], "dtype": "float32"}),
            node(
                "conv",
                "conv.conv2d",
                {
                    "filters": 4,
                    "kernel_size": [3, 3],
                    "strides": [2, 2],
                    "padding": "same",
                    "data_format": "channels_last",
                },
            ),
            node(
                "pool",
                "conv.max_pool2d",
                {
                    "pool_size": [2, 2],
                    "strides": [2, 2],
                    "padding": "same",
                    "data_format": "channels_last",
                },
            ),
            node("output", "output.custom"),
        ],
        edges=[
            edge("e1", "image", "conv"),
            edge("e2", "conv", "pool"),
            edge("e3", "pool", "output"),
        ],
    )
    torch, model = build_model(graph)
    result = model(torch.randn(2, 7, 9, 3))
    assert tuple(result.shape) == (2, 2, 3, 4)

###############################################################################
@pytest.mark.parametrize("definition_id", ["recurrent.lstm", "recurrent.gru"])
def test_generated_pytorch_recurrent_layers_derive_input_size_from_shape(
    definition_id: str,
) -> None:
    graph = ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [None, 5, 6], "dtype": "float32"}),
            node("recurrent", definition_id, {"units": 3, "return_sequences": False}),
            node("output", "output.regression"),
        ],
        edges=[edge("e1", "input", "recurrent"), edge("e2", "recurrent", "output")],
    )
    torch, model = build_model(graph)
    result = model(torch.randn(2, 5, 6))
    assert tuple(result.shape) == (2, 3)

###############################################################################
def test_generated_pytorch_embedding_and_normalization_execute() -> None:
    embedding_graph = ModelGraph(
        nodes=[
            node("input", "input.tensor", {"shape": [None, 4], "dtype": "int64"}),
            node("embedding", "layer.embedding", {"vocabulary_size": 20, "embedding_dim": 5}),
            node("output", "output.custom"),
        ],
        edges=[edge("e1", "input", "embedding"), edge("e2", "embedding", "output")],
    )
    torch, embedding_model = build_model(embedding_graph)
    embedding_result = embedding_model(torch.randint(0, 20, (2, 4), dtype=torch.int64))
    assert tuple(embedding_result.shape) == (2, 4, 5)

    normalization_graph = ModelGraph(
        nodes=[
            node("input", "input.image", {"shape": [None, 4, 4, 3], "dtype": "float32"}),
            node("batch", "layer.batch_norm", {"axis": -1}),
            node("layer", "layer.layer_norm", {"axis": [-1]}),
            node("output", "output.custom"),
        ],
        edges=[
            edge("e1", "input", "batch"),
            edge("e2", "batch", "layer"),
            edge("e3", "layer", "output"),
        ],
    )
    _, normalization_model = build_model(normalization_graph)
    normalization_result = normalization_model(torch.randn(2, 4, 4, 3))
    assert tuple(normalization_result.shape) == (2, 4, 4, 3)

###############################################################################
def test_generated_pytorch_common_merges_execute() -> None:
    graph = ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("left", "layer.dense", {"units": 3}),
            node("right", "layer.dense", {"units": 3}),
            node("merge", "merge.add"),
            node("output", "output.regression"),
        ],
        edges=[
            edge("e1", "input", "left"),
            edge("e2", "input", "right"),
            edge("e3", "left", "merge", "inputs"),
            edge("e4", "right", "merge", "inputs"),
            edge("e5", "merge", "output"),
        ],
    )
    torch, model = build_model(graph)
    result = model(torch.randn(2, 4))
    assert tuple(result.shape) == (2, 3)
