from __future__ import annotations

from neuro_weave.catalog.loader import get_node_catalog
from neuro_weave.compilation.compiler import compile_to_ir
from neuro_weave.domain.enums import TargetFramework
from neuro_weave.domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position


def test_compiler_generates_unique_keyword_safe_variable_names() -> None:
    graph = ModelGraph(
        nodes=[
            NodeInstance(id="input", definition_id="input.tabular", label="class", position=Position(x=0, y=0), parameters={"shape": [None, 4], "dtype": "float32"}),
            NodeInstance(id="dense-a", definition_id="layer.dense", label="hidden", position=Position(x=1, y=0), parameters={"units": 4}),
            NodeInstance(id="dense-b", definition_id="layer.dense", label="hidden", position=Position(x=2, y=0), parameters={"units": 2}),
            NodeInstance(id="output", definition_id="output.classification", label="result", position=Position(x=3, y=0), parameters={}),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="dense-a", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="dense-a", source_port_id="output", target_node_id="dense-b", target_port_id="input"),
            EdgeInstance(id="e3", source_node_id="dense-b", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    ir = compile_to_ir(graph, TargetFramework.KERAS, get_node_catalog())
    names = [node.variable_name for node in ir.ordered_nodes]
    assert names == ["node_class", "hidden", "hidden_2", "result"]
