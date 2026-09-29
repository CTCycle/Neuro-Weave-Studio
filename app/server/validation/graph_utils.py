from __future__ import annotations

from collections import defaultdict, deque

from domain.graph import EdgeInstance, ModelGraph
from domain.manifests import NodeManifest
from domain.ports import materialized_inputs


###############################################################################
def adjacency(graph: ModelGraph) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        result[edge.source_node_id].append(edge.target_node_id)
    return result

###############################################################################
def topological_order(graph: ModelGraph) -> list[str]:
    node_ids = {node.id for node in graph.nodes}
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.source_node_id in node_ids and edge.target_node_id in node_ids:
            outgoing[edge.source_node_id].append(edge.target_node_id)
            indegree[edge.target_node_id] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    ordered: list[str] = []
    while queue:
        current = queue.popleft()
        ordered.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(ordered) != len(node_ids):
        raise ValueError("Graph contains a cycle")
    return ordered

###############################################################################
def would_create_cycle(graph: ModelGraph, source_node_id: str, target_node_id: str) -> bool:
    if source_node_id == target_node_id:
        return True
    outgoing = adjacency(graph)
    stack = [target_node_id]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current == source_node_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        stack.extend(outgoing.get(current, []))
    return False

###############################################################################
def incoming_edges(graph: ModelGraph) -> dict[str, list[EdgeInstance]]:
    result: dict[str, list[EdgeInstance]] = defaultdict(list)
    for edge in graph.edges:
        result[edge.target_node_id].append(edge)
    return result

###############################################################################
def reachable_nodes(graph: ModelGraph, starts: set[str]) -> set[str]:
    """Return nodes reachable by following graph edges from ``starts``."""

    outgoing = adjacency(graph)
    visited: set[str] = set()
    stack = list(starts)
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(outgoing.get(current, []))
    return visited

###############################################################################
def nodes_reaching(graph: ModelGraph, targets: set[str]) -> set[str]:
    """Return nodes that can reach one of ``targets`` by following edges."""

    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        reverse[edge.target_node_id].append(edge.source_node_id)
    visited: set[str] = set()
    stack = list(targets)
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(reverse.get(current, []))
    return visited

###############################################################################
def ordered_incoming_edges(
    graph: ModelGraph,
    node_id: str,
    manifest: NodeManifest,
    parameters: dict[str, object] | None = None,
) -> list[EdgeInstance]:
    """Return inbound edges in the node contract's port order.

    Named ports use the manifest order so, for example, an attention node
    always sees ``query`` before ``context`` even when the user connected them
    in the opposite order. Multiple edges sharing one ``tensor_list`` port
    have no named-port order; preserve their graph order so ordered operations
    such as matrix multiplication retain the user's operand order.
    """

    port_order = {
        port.id: index
        for index, port in enumerate(materialized_inputs(manifest, parameters))
    }
    return [
        edge
        for _, edge in sorted(
            enumerate(incoming_edges(graph).get(node_id, [])),
            key=lambda indexed_edge: (
                port_order.get(indexed_edge[1].target_port_id, len(port_order)),
                indexed_edge[1].target_port_id,
                indexed_edge[0],
            ),
        )
    ]
