from __future__ import annotations

from typing import Any

from domain.compilation_plan import normalize_compilation_plan

###############################################################################
def _remove_detection_placeholder(raw: dict[str, Any]) -> None:
    removed = {
        str(node.get("id"))
        for node in raw.get("nodes", [])
        if isinstance(node, dict) and node.get("definition_id") == "output.detection_placeholder"
    }
    if not removed:
        return
    raw["nodes"] = [
        node
        for node in raw.get("nodes", [])
        if not isinstance(node, dict) or node.get("id") not in removed
    ]
    raw["edges"] = [
        edge
        for edge in raw.get("edges", [])
        if not isinstance(edge, dict)
        or (edge.get("source_node_id") not in removed and edge.get("target_node_id") not in removed)
    ]
    raw["output_bindings"] = [
        binding
        for binding in raw.get("output_bindings", [])
        if not isinstance(binding, dict) or binding.get("node_id") not in removed
    ]

###############################################################################
def normalize_graph_payload(value: Any) -> dict[str, Any]:
    """Convert graph schema 5 payloads to the canonical schema 6 payload.

    This is deliberately the only compatibility boundary. Domain models and
    services receive the normalized representation and never branch on the
    incoming schema version.
    """

    raw = dict(value) if isinstance(value, dict) else {}
    version = raw.get("schema_version", 5)
    if version not in {5, 6}:
        raise ValueError(f"Unsupported graph schema version: {version}")
    _remove_detection_placeholder(raw)
    raw["schema_version"] = 6
    configuration = raw.get("model_configuration")
    if isinstance(configuration, dict):
        normalized = dict(configuration)
        keras = normalized.get("keras", normalized)
        native = normalized.get("native", {})
        if "keras" not in normalized:
            normalized = {"keras": keras, "native": native}
        normalized["keras"] = dict(normalized.get("keras", {}))
        normalized["keras"]["compilation"] = normalize_compilation_plan(
            normalized["keras"].get("compilation", normalized["keras"])
        ).model_dump(mode="json")
        normalized["native"] = dict(normalized.get("native", {}))
        for backend in ("tensorflow", "pytorch", "jax"):
            branch = dict(normalized["native"].get(backend, {}))
            branch["compilation"] = normalize_compilation_plan(
                branch.get("compilation", branch)
            ).model_dump(mode="json")
            normalized["native"][backend] = branch
        raw["model_configuration"] = normalized
    return raw

###############################################################################
def normalize_bundle_payload(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, dict) else {}
    version = raw.get("format_version", 6)
    if version not in {6, 7}:
        raise ValueError(f"Unsupported project bundle format: {version}")
    raw["format_version"] = 7
    if isinstance(raw.get("project"), dict):
        project = dict(raw["project"])
        project["graph"] = normalize_graph_payload(project.get("graph", {}))
        raw["project"] = project
    return raw
