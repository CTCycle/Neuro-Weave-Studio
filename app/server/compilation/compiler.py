from __future__ import annotations

import keyword
import re

from common.catalogs.loader import NodeCatalog
from domain.compilation import CanonicalModelIR, CompiledNode
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph, ShapeSummary
from validation.graph_utils import ordered_incoming_edges, topological_order
from validation.shape_inference import infer_shapes

###############################################################################
def _variable_name(node_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
    if value and (value[0].isdigit() or keyword.iskeyword(value)):
        value = f"node_{value}"
    return value or "node"

###############################################################################
def compile_to_ir(
    graph: ModelGraph,
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
    catalog: NodeCatalog,
    shape_summaries: list[ShapeSummary] | None = None,
) -> CanonicalModelIR:
    nodes = {node.id: node for node in graph.nodes}
    ordered_nodes: list[CompiledNode] = []
    used_variable_names: set[str] = set()
    for node_id in topological_order(graph):
        node = nodes[node_id]
        manifest = catalog.get(node.definition_id, node.definition_version)
        input_node_ids = [
            edge.source_node_id
            for edge in ordered_incoming_edges(graph, node_id, manifest)
        ]
        variable_name = _variable_name(node.label or node.id)
        base_name = variable_name
        suffix = 2
        while variable_name in used_variable_names:
            variable_name = f"{base_name}_{suffix}"
            suffix += 1
        used_variable_names.add(variable_name)
        ordered_nodes.append(
            CompiledNode(
                node_id=node_id,
                definition_id=node.definition_id,
                variable_name=variable_name,
                input_node_ids=input_node_ids,
                parameters=node.parameters,
            )
        )
    return CanonicalModelIR(
        execution_backend=execution_backend,
        keras_enabled=keras_enabled,
        ordered_nodes=ordered_nodes,
        input_node_ids=[node.id for node in graph.nodes if node.definition_id.startswith("input.")],
        output_node_ids=[node.id for node in graph.nodes if node.definition_id.startswith("output.")],
        shape_summaries=(shape_summaries if shape_summaries is not None else infer_shapes(graph, catalog)),
    )
