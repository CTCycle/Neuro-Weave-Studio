from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from common.catalogs.loader import NodeCatalog
from domain.graph import ModelGraph, NodeInstance, OutputBinding, ShapeSummary
from domain.manifests import NodeManifest, PortDefinition
from domain.tensor import TensorSpec


###############################################################################
@dataclass(frozen=True)
class ResolvedOutput:
    """An output binding with the catalog metadata needed by validators/exporters."""

    binding: OutputBinding
    name: str
    manifest: NodeManifest | None
    node: NodeInstance | None
    port: PortDefinition | None
    explicit: bool


###############################################################################
def input_nodes(graph: ModelGraph, catalog: NodeCatalog) -> list[NodeInstance]:
    result: list[NodeInstance] = []
    for node in graph.nodes:
        try:
            manifest = catalog.get(node.definition_id, node.definition_version)
        except KeyError:
            continue
        if manifest.semantic_role == "input":
            result.append(node)
    return result


###############################################################################
def _condition_matches(condition: dict[str, object], parameters: dict[str, object]) -> bool:
    for parameter_id, expected in condition.items():
        actual = parameters.get(parameter_id)
        if isinstance(expected, dict):
            if "equals" in expected and actual != expected["equals"]:
                return False
            if "not_equals" in expected and actual == expected["not_equals"]:
                return False
            if "in" in expected and actual not in expected["in"]:
                return False
            if "not_in" in expected and actual in expected["not_in"]:
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _tensor_output_ports(
    manifest: NodeManifest, parameters: dict[str, object] | None = None
) -> Iterable[PortDefinition]:
    effective_parameters = parameters or {}
    return (
        port
        for port in manifest.outputs
        if _condition_matches(port.enabled_when, effective_parameters)
        if port.kind.value in {"tensor", "tensor_list", "mask", "scalar"}
    )


def _output_name(node: NodeInstance | None, manifest: NodeManifest | None, port: PortDefinition | None, fallback: str) -> str:
    base_name = node.label or node.id if node is not None and node.label else node.id if node is not None else fallback
    if node is None or manifest is None or port is None:
        return base_name
    enabled_port_ids = [item.id for item in _tensor_output_ports(manifest, node.parameters)]
    return base_name if len(enabled_port_ids) == 1 else f"{base_name}_{port.id}"


###############################################################################
def inferred_output_bindings(graph: ModelGraph, catalog: NodeCatalog) -> list[OutputBinding]:
    """Infer every unconsumed output port on a non-input node.

    A terminal is a port, rather than only a node. This keeps the graph model
    useful for branches and for future multi-output node definitions.
    """

    connected_ports = {
        (edge.source_node_id, edge.source_port_id) for edge in graph.edges
    }
    bindings: list[OutputBinding] = []
    for node in graph.nodes:
        try:
            manifest = catalog.get(node.definition_id, node.definition_version)
        except KeyError:
            continue
        if manifest.semantic_role == "input":
            continue
        for port in _tensor_output_ports(manifest, node.parameters):
            if (node.id, port.id) not in connected_ports:
                bindings.append(OutputBinding(node_id=node.id, port_id=port.id))
    return bindings


###############################################################################
def output_bindings(graph: ModelGraph, catalog: NodeCatalog) -> tuple[list[OutputBinding], bool]:
    """Return explicit bindings when present, otherwise inferred terminals."""

    if graph.output_bindings:
        return list(graph.output_bindings), True
    return inferred_output_bindings(graph, catalog), False


###############################################################################
def resolved_outputs(graph: ModelGraph, catalog: NodeCatalog) -> list[ResolvedOutput]:
    bindings, explicit = output_bindings(graph, catalog)
    nodes = {node.id: node for node in graph.nodes}
    result: list[ResolvedOutput] = []
    for binding in bindings:
        node = nodes.get(binding.node_id)
        manifest: NodeManifest | None = None
        port: PortDefinition | None = None
        if node is not None:
            try:
                manifest = catalog.get(node.definition_id, node.definition_version)
            except KeyError:
                manifest = None
            if manifest is not None:
                port = next((item for item in manifest.outputs if item.id == binding.port_id), None)
        result.append(
            ResolvedOutput(
                binding=binding,
                name=_output_name(node, manifest, port, binding.node_id),
                manifest=manifest,
                node=node,
                port=port,
                explicit=explicit,
            )
        )
    return result


###############################################################################
def _head_node(graph: ModelGraph, output: ResolvedOutput) -> NodeInstance | None:
    """Find the computational node that most directly determines an output."""

    nodes = {node.id: node for node in graph.nodes}
    current = output.node
    incoming_by_port = {
        (edge.target_node_id, edge.target_port_id): edge.source_node_id for edge in graph.edges
    }
    visited: set[str] = set()
    while current is not None and current.id not in visited:
        visited.add(current.id)
        if output.manifest is None or output.manifest.semantic_role != "output_adapter":
            return current
        source_id = incoming_by_port.get((current.id, "input"))
        current = nodes.get(source_id) if source_id is not None else None
        output = ResolvedOutput(
            binding=OutputBinding(node_id=current.id, port_id="output")
            if current is not None
            else output.binding,
            name=current.label if current is not None and current.label else current.id
            if current is not None
            else output.name,
            manifest=None,
            node=current,
            port=None,
            explicit=output.explicit,
        )
    return current


###############################################################################
def _head_activation_and_width(
    graph: ModelGraph, output: ResolvedOutput, shape: TensorSpec | None
) -> tuple[str | None, int | None, str | None, list[str]]:
    head = _head_node(graph, output)
    if head is None:
        return None, None, None, []
    parameters = head.parameters
    activation = parameters.get("activation")
    width_value = parameters.get("units", parameters.get("filters"))
    width = width_value if isinstance(width_value, int) and not isinstance(width_value, bool) else None
    head_kind = head.definition_id
    evidence = [f"head={head.definition_id}"]
    if isinstance(activation, str):
        evidence.append(f"activation={activation}")
    if width is not None:
        evidence.append(f"width={width}")
    if width is None and shape is not None and shape.shape:
        dimension = shape.shape[-1]
        width = dimension if isinstance(dimension, int) else None
        if width is not None:
            evidence.append(f"output_width={width}")
    return activation if isinstance(activation, str) else None, width, head_kind, evidence


###############################################################################
def infer_model_type(
    graph: ModelGraph,
    catalog: NodeCatalog,
    shape_summaries: list[ShapeSummary] | None = None,
) -> tuple[str, str, list[str]]:
    """Infer a non-binding task hint without constraining graph execution."""

    outputs = resolved_outputs(graph, catalog)
    if not outputs:
        return "generic", "low", ["no valid output binding is available"]
    if len(outputs) > 1:
        return "multi_output", "high", [f"{len(outputs)} outputs are selected"]

    output = outputs[0]
    shape = None
    if shape_summaries is not None:
        summary = next((item for item in shape_summaries if item.node_id == output.binding.node_id), None)
        if summary is not None:
            shape = summary.outputs.get(output.binding.port_id)
    activation, width, head_kind, evidence = _head_activation_and_width(graph, output, shape)
    task_hint = output.manifest.task_hint if output.manifest is not None else None
    if task_hint == "classification":
        if activation == "sigmoid" and width is not None and width > 1:
            return "multilabel_classification", "medium", [*evidence, "classification output adapter"]
        if width == 1:
            return "binary_classification", "high", [*evidence, "classification output adapter"]
        if width is not None and width >= 2:
            return "multiclass_classification", "high", [*evidence, "classification output adapter"]
        return "generic", "low", [*evidence, "classification output adapter", "class width is unknown"]
    if task_hint == "regression":
        return "regression", "high", [*evidence, "regression output adapter"]
    if task_hint == "sequence":
        return "sequence", "high", [*evidence, "sequence output adapter"]
    if task_hint == "embedding" or head_kind == "layer.embedding":
        return "embedding", "high", [*evidence, "embedding head"]
    if activation == "sigmoid" and width == 1:
        return "binary_classification", "medium", evidence
    if activation == "sigmoid" and width is not None and width > 1:
        return "multilabel_classification", "medium", evidence
    if activation == "softmax" and width is not None and width >= 2:
        return "multiclass_classification", "medium", evidence
    if shape is not None and shape.rank == 3 and head_kind in {
        "recurrent.lstm",
        "recurrent.gru",
        "recurrent.simple_rnn",
    }:
        return "sequence", "low", [*evidence, "rank-3 recurrent output"]
    return "generic", "low", [*evidence, "graph does not uniquely identify a task"]
