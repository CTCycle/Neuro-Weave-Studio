from __future__ import annotations

from collections import defaultdict, deque

from neuro_weave.domain.graph import EdgeInstance, ModelGraph


def adjacency(graph: ModelGraph) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        result[edge.source_node_id].append(edge.target_node_id)
    return result


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


def incoming_edges(graph: ModelGraph) -> dict[str, list[EdgeInstance]]:
    result: dict[str, list[EdgeInstance]] = defaultdict(list)
    for edge in graph.edges:
        result[edge.target_node_id].append(edge)
    return result
