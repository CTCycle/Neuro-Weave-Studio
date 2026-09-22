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
def edge(
    edge_id: str,
    source: str,
    target: str,
    target_port: str = "input",
) -> EdgeInstance:
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
    validation = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        False,
    )
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
def test_generated_pytorch_conv_pool_preserves_channels_last_contract() -> None:
    graph = ModelGraph(
        nodes=[
            node(
                "image",
                "input.image",
                {"shape": [7, 9, 3], "dtype": "float32"},
            ),
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
def test_generated_pytorch_lstm_derives_feature_width_from_inferred_shape() -> None:
    graph = ModelGraph(
        nodes=[
            node(
                "input",
                "input.sequence",
                {"shape": [5, 6], "dtype": "float32"},
            ),
            node(
                "recurrent",
                "recurrent.lstm",
                {"units": 3, "return_sequences": False},
            ),
            node("output", "output.regression"),
        ],
        edges=[
            edge("e1", "input", "recurrent"),
            edge("e2", "recurrent", "output"),
        ],
    )

    torch, model = build_model(graph)
    result = model(torch.randn(2, 5, 6))

    assert tuple(result.shape) == (2, 3)
