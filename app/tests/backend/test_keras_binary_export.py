from pathlib import Path

import pytest

from neuro_weave.domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from neuro_weave.exporters.keras_exporter import KerasBinaryExporter


def test_uninitialized_keras_export_when_dependency_available(tmp_path: Path) -> None:
    pytest.importorskip("keras")
    graph = ModelGraph(
        nodes=[
            NodeInstance(id="input", definition_id="input.tabular", position=Position(x=0, y=0), parameters={"shape": [None, 4], "dtype": "float32"}),
            NodeInstance(id="dense", definition_id="layer.dense", position=Position(x=1, y=0), parameters={"units": 2, "activation": "softmax", "use_bias": True}),
            NodeInstance(id="output", definition_id="output.classification", position=Position(x=2, y=0), parameters={}),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="dense", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="dense", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    destination = KerasBinaryExporter().export(graph, tmp_path / "model.keras")
    assert destination.is_file()
