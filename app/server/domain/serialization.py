from __future__ import annotations

from typing import Any, cast

from domain.compilation_plan import normalize_compilation_plan


###############################################################################
def _remove_detection_placeholder(raw: dict[str, object]) -> None:
    nodes = cast(list[object], raw.get("nodes", [])) if isinstance(raw.get("nodes", []), list) else []
    removed = {
        str(cast(dict[str, object], node).get("id"))
        for node in nodes
        if isinstance(node, dict)
        and cast(dict[str, object], node).get("definition_id") == "output.detection_placeholder"
    }
    if not removed:
        return
    raw["nodes"] = [
        node
        for node in nodes
        if not isinstance(node, dict) or cast(dict[str, object], node).get("id") not in removed
    ]
    edges = cast(list[object], raw.get("edges", [])) if isinstance(raw.get("edges", []), list) else []
    raw["edges"] = [
        edge
        for edge in edges
        if not isinstance(edge, dict)
        or (
            cast(dict[str, object], edge).get("source_node_id") not in removed
            and cast(dict[str, object], edge).get("target_node_id") not in removed
        )
    ]
    output_bindings = (
        cast(list[object], raw.get("output_bindings", []))
        if isinstance(raw.get("output_bindings", []), list)
        else []
    )
    raw["output_bindings"] = [
        binding
        for binding in output_bindings
        if not isinstance(binding, dict)
        or cast(dict[str, object], binding).get("node_id") not in removed
    ]

###############################################################################
def normalize_graph_payload(value: Any) -> dict[str, Any]:
    """Convert graph schema 5 payloads to the canonical schema 6 payload.

    This is deliberately the only compatibility boundary. Domain models and
    services receive the normalized representation and never branch on the
    incoming schema version.
    """

    raw = dict(cast(dict[str, object], value)) if isinstance(value, dict) else {}
    version = raw.get("schema_version", 5)
    if version not in {5, 6}:
        raise ValueError(f"Unsupported graph schema version: {version}")
    _remove_detection_placeholder(raw)
    raw["schema_version"] = 6
    configuration = raw.get("model_configuration")
    if isinstance(configuration, dict):
        normalized = dict(cast(dict[str, object], configuration))
        keras = normalized.get("keras", normalized)
        native = cast(dict[str, object], normalized.get("native", {}))
        if "keras" not in normalized:
            normalized = {"keras": keras, "native": native}
        keras_branch = dict(cast(dict[str, object], normalized.get("keras", {})))
        keras_branch["compilation"] = normalize_compilation_plan(
            keras_branch.get("compilation", keras_branch)
        ).model_dump(mode="json")
        normalized["keras"] = keras_branch
        for backend in ("tensorflow", "pytorch", "jax"):
            branch = dict(cast(dict[str, object], native.get(backend, {})))
            branch["compilation"] = normalize_compilation_plan(
                branch.get("compilation", branch)
            ).model_dump(mode="json")
            native[backend] = branch
        normalized["native"] = native
        raw["model_configuration"] = normalized
    return cast(dict[str, Any], raw)

###############################################################################
def normalize_bundle_payload(value: Any) -> dict[str, Any]:
    raw = dict(cast(dict[str, object], value)) if isinstance(value, dict) else {}
    version = raw.get("format_version", 6)
    if version not in {6, 7}:
        raise ValueError(f"Unsupported project bundle format: {version}")
    raw["format_version"] = 7
    if isinstance(raw.get("project"), dict):
        project = dict(cast(dict[str, object], raw["project"]))
        project["graph"] = normalize_graph_payload(project.get("graph", {}))
        raw["project"] = project
    return cast(dict[str, Any], raw)
