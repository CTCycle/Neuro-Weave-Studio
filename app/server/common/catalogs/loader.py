from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from domain.manifests import NodeManifest

RESOURCE_ROOT = Path(__file__).resolve().parents[2] / "resources" / "nodes"
CATALOG_EXTENSION_PATH = RESOURCE_ROOT.parent / "catalog_extensions.json"
PARAMETER_SURFACE_PATH = RESOURCE_ROOT.parent / "parameter_surfaces.json"


###############################################################################
def _expand_parameter_groups(
    raw: dict[str, Any], parameter_groups: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Expand resource-declared parameter groups before normalization."""

    item = dict(raw)
    references = item.get("parameter_groups", [])
    if references is None:
        references = []
    if not isinstance(references, list) or any(
        not isinstance(reference, str) for reference in references
    ):
        raise RuntimeError(f"Parameter groups for {item.get('id', '<unknown>')} must be a list")
    expanded: list[dict[str, Any]] = []
    seen_parameters: set[str] = set()
    seen_aliases: set[str] = set()
    for reference in references:
        group = parameter_groups.get(reference)
        if group is None:
            raise RuntimeError(
                f"Unknown parameter group '{reference}' for {item.get('id', '<unknown>')}"
            )
        for parameter in group:
            identifier = str(parameter.get("id", ""))
            if not identifier:
                raise RuntimeError(
                    f"Parameter group '{reference}' contains an empty identifier for "
                    f"{item.get('id', '<unknown>')}"
                )
            if identifier in seen_aliases:
                raise RuntimeError(
                    f"Parameter alias collision for {item.get('id', '<unknown>')}: {identifier}"
                )
            if identifier in seen_parameters:
                raise RuntimeError(
                    f"Parameter group collision for {item.get('id', '<unknown>')}: {identifier}"
                )
            aliases = parameter.get("aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
                raise RuntimeError(
                    f"Aliases for {item.get('id', '<unknown>')}:{identifier} must be a list"
                )
            if seen_parameters.intersection(aliases):
                raise RuntimeError(
                    f"Parameter alias collision for {item.get('id', '<unknown>')}: {identifier}"
                )
            seen_parameters.add(identifier)
            seen_parameters.update(aliases)
            seen_aliases.update(aliases)
            expanded.append(dict(parameter))
    explicit = item.get("parameters", [])
    if explicit is None:
        explicit = []
    if not isinstance(explicit, list):
        raise RuntimeError(f"Parameters for {item.get('id', '<unknown>')} must be a list")
    for parameter in cast(list[dict[str, Any]], explicit):
        identifier = str(parameter.get("id", ""))
        if not identifier:
            raise RuntimeError(f"Parameter for {item.get('id', '<unknown>')} must have an id")
        aliases = parameter.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise RuntimeError(
                f"Aliases for {item.get('id', '<unknown>')}:{identifier} must be a list"
            )
        if seen_aliases.intersection({identifier, *aliases}):
            raise RuntimeError(
                f"Parameter alias collision for {item.get('id', '<unknown>')}: {identifier}"
            )
        if seen_parameters.intersection({identifier, *aliases}):
            raise RuntimeError(
                f"Parameter surface collision for {item.get('id', '<unknown>')}: {identifier}"
            )
        seen_parameters.add(identifier)
        seen_parameters.update(aliases)
    item["parameters"] = [*expanded, *cast(list[dict[str, Any]], explicit)]
    return item

###############################################################################
def _enrich_manifest_parameters(
    raw: dict[str, Any],
    parameter_groups: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Expand the declarative catalog into the inspector contract."""

    item = _expand_parameter_groups(raw, parameter_groups or {})
    node_id = str(item.get("id", ""))
    item["parameters"] = [dict(parameter) for parameter in item.get("parameters", [])]
    category = str(item.get("category", ""))
    if category == "inputs":
        item.setdefault("semantic_role", "input")
        if node_id == "input.constant":
            item.setdefault("source_kind", "constant")
        elif node_id == "input.parameter":
            item.setdefault("source_kind", "trainable_parameter")
        else:
            item.setdefault("source_kind", "runtime_input")
    elif category == "outputs":
        item.setdefault("semantic_role", "output_adapter")
        item.setdefault("task_hint", _output_task_hint(node_id))
    else:
        item.setdefault("semantic_role", "operation")
    return item

###############################################################################
def _output_task_hint(node_id: str) -> str | None:
    return {
        "output.classification": "classification",
        "output.regression": "regression",
        "output.segmentation": "segmentation",
        "output.sequence": "sequence",
        "output.multi_head": "multi_output",
        "output.detection": "detection",
        "output.custom": "generic",
    }.get(node_id)

###############################################################################
def _merge_manifest_data(
    raw: dict[str, Any],
    override: dict[str, Any] | None,
    shared_parameter_capabilities: dict[str, dict[str, Any]] | None = None,
    parameter_groups: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Apply checked-in declarative capability/operation data to a node."""

    item = dict(raw)
    raw_parameters = raw.get("parameters", [])
    parameter_items = [*cast(list[Any], raw_parameters)] if isinstance(raw_parameters, list) else []
    for group_id in cast(list[Any], raw.get("parameter_groups", [])):
        if not isinstance(group_id, str):
            continue
        parameter_items.extend((parameter_groups or {}).get(group_id, []))
    known_parameter_ids = {
        str(cast(dict[str, Any], parameter).get("id", ""))
        for parameter in parameter_items
        if isinstance(parameter, dict)
    }
    implementations = dict(item.get("implementations", {}))
    for implementation, capabilities in (shared_parameter_capabilities or {}).items():
        mapping = dict(implementations.get(implementation, {}))
        parameters = dict(mapping.get("parameters", {}))
        parameters.update(
            {
                parameter_id: capability
                for parameter_id, capability in capabilities.items()
                if parameter_id in known_parameter_ids
            }
        )
        if parameters:
            mapping["parameters"] = parameters
        implementations[implementation] = mapping
    if implementations:
        item["implementations"] = implementations
    if not override:
        return item
    for key, value in override.items():
        if key != "implementations":
            item[key] = value
            continue
        implementations = dict(item.get("implementations", {}))
        for implementation, raw_mapping in value.items():
            mapping = dict(implementations.get(implementation, {}))
            for mapping_key, mapping_value in raw_mapping.items():
                if mapping_key in {"parameters", "variant_symbols"}:
                    merged = dict(mapping.get(mapping_key, {}))
                    merged.update(mapping_value)
                    mapping[mapping_key] = merged
                else:
                    mapping[mapping_key] = mapping_value
            implementations[implementation] = mapping
        item["implementations"] = implementations
    return item

###############################################################################
def _expand_operation_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Expand compact declarative operation metadata into a normal manifest."""

    family = str(spec["family"])
    variadic = bool(spec.get("variadic", False))
    accepts_multiple = variadic or family in {"binary", "elementwise"}
    input_definition = {
        "id": "inputs" if accepts_multiple else "input",
        "display_name": "Inputs" if accepts_multiple else "Input",
        "kind": "tensor_list" if accepts_multiple else "tensor",
        "required": True,
        "accepts_multiple": accepts_multiple,
        "rank": None,
    }
    shape_behavior = str(
        spec.get("shape_behavior", "reduce" if family == "reduction" else "preserve")
    )
    shape_rule = "concatenate" if shape_behavior in {"concatenate", "stack"} else "operation"
    implementations: dict[str, dict[str, Any]] = {}
    symbols = cast(dict[str, str], spec["symbols"])
    for implementation, symbol in symbols.items():
        mapping: dict[str, Any] = {"supported": True, "symbol": symbol}
        if spec.get("adapter"):
            mapping["adapter"] = spec["adapter"]
        if implementation == "keras":
            mapping["execution_backends"] = {
                "tensorflow": True,
                "pytorch": True,
                "jax": True,
            }
        implementations[implementation] = mapping
    return {
        "id": spec["id"],
        "version": 1,
        "display_name": spec["display_name"],
        "category": spec.get("category", "mathematical"),
        "description": spec.get("description", f"{spec['display_name']} operation."),
        "inputs": [input_definition],
        "outputs": [
            {
                "id": "output",
                "display_name": "Output",
                "kind": "tensor",
                "required": True,
                "accepts_multiple": False,
                "rank": None,
            }
        ],
        "parameters": spec.get("parameters", []),
        "shape_rule": {"kind": shape_rule, "config": {}},
        "operation": {
            "family": family,
            "shape_behavior": shape_behavior,
            "output_dtype": spec.get("output_dtype", "same"),
            "arity": int(spec.get("arity", 1)),
            "variadic": variadic,
            "arguments": spec.get("arguments", []),
            "keywords": spec.get("keywords", []),
        },
        "implementations": implementations,
        "ui": {
            "icon": spec.get("icon", "Sigma"),
            "accent": spec.get("accent", "amber"),
            "width": 250,
            "height": 120,
        },
        "deprecated": False,
        "documentation_ref": "assets/docs/nodes/mathematical_operations.md",
    }

###############################################################################
def _catalog_extensions() -> dict[str, Any]:
    if not CATALOG_EXTENSION_PATH.exists():
        return {"nodes": [], "overrides": {}}
    raw = json.loads(CATALOG_EXTENSION_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Catalog extension data must be an object")
    return cast(dict[str, Any], raw)

###############################################################################
def _parameter_surface_catalog() -> dict[str, list[dict[str, Any]]]:
    """Load node-specific public parameters from the declarative resource."""

    if not PARAMETER_SURFACE_PATH.exists():
        return {}
    raw = json.loads(PARAMETER_SURFACE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), dict):
        raise RuntimeError("parameter_surfaces.json must contain a nodes object")
    surfaces: dict[str, list[dict[str, Any]]] = {}
    for node_id, parameters in raw["nodes"].items():
        if not isinstance(node_id, str) or not isinstance(parameters, list):
            raise RuntimeError("Each declarative parameter surface must be a list")
        if any(not isinstance(parameter, dict) for parameter in parameters):
            raise RuntimeError(f"Parameter surface for {node_id} contains a non-object")
        surfaces[node_id] = [dict(parameter) for parameter in parameters]
    return surfaces

###############################################################################
def _parameter_group_catalog(
    extensions: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    raw_groups = extensions.get("parameter_groups", {})
    if not isinstance(raw_groups, dict):
        raise RuntimeError("parameter_groups must be an object")
    groups: dict[str, list[dict[str, Any]]] = {}
    for group_id, values in raw_groups.items():
        if not isinstance(group_id, str) or not isinstance(values, list):
            raise RuntimeError("Each parameter group must be a list of definitions")
        if any(not isinstance(value, dict) for value in values):
            raise RuntimeError(f"Parameter group '{group_id}' contains a non-object definition")
        groups[group_id] = [dict(value) for value in values]
    raw_assignments = extensions.get("parameter_group_assignments", {})
    if not isinstance(raw_assignments, dict):
        raise RuntimeError("parameter_group_assignments must be an object")
    assignments: dict[str, list[str]] = {}
    for node_id, references in raw_assignments.items():
        if not isinstance(node_id, str) or not isinstance(references, list):
            raise RuntimeError("Each parameter group assignment must be a list")
        assignments[node_id] = [str(reference) for reference in references]
    return groups, assignments

###############################################################################
def _combined_override(
    capability: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not capability and not override:
        return None
    result = dict(capability or {})
    for key, value in (override or {}).items():
        if key != "implementations":
            result[key] = value
            continue
        implementations = dict(result.get("implementations", {}))
        for implementation, mapping in value.items():
            existing = dict(implementations.get(implementation, {}))
            for mapping_key, mapping_value in mapping.items():
                if mapping_key in {"parameters", "variant_symbols"}:
                    merged = dict(existing.get(mapping_key, {}))
                    merged.update(mapping_value)
                    existing[mapping_key] = merged
                else:
                    existing[mapping_key] = mapping_value
            implementations[implementation] = existing
        result["implementations"] = implementations
    return result

###############################################################################
class NodeCatalog:

    # -------------------------------------------------------------------------
    def __init__(self, manifests: list[NodeManifest]) -> None:
        self._manifests = sorted(manifests, key=lambda item: (item.category, item.display_name))
        self._by_key = {(item.id, item.version): item for item in self._manifests}

    # -------------------------------------------------------------------------
    @property
    def manifests(self) -> list[NodeManifest]:
        return list(self._manifests)

    # -------------------------------------------------------------------------
    def get(self, definition_id: str, version: int = 1) -> NodeManifest:
        try:
            return self._by_key[(definition_id, version)]
        except KeyError as exc:
            raise KeyError(f"Unknown node definition: {definition_id}@{version}") from exc

###############################################################################
def get_node_catalog() -> NodeCatalog:
    extensions = _catalog_extensions()
    parameter_surfaces = _parameter_surface_catalog()
    parameter_groups, group_assignments = _parameter_group_catalog(extensions)
    overrides = cast(dict[str, dict[str, Any]], extensions.get("overrides", {}))
    capabilities = cast(dict[str, dict[str, Any]], extensions.get("capabilities", {}))
    shared_parameter_capabilities = cast(
        dict[str, dict[str, Any]], extensions.get("shared_parameter_capabilities", {})
    )
    manifests: list[NodeManifest] = []
    for path in sorted(RESOURCE_ROOT.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = cast(list[dict[str, Any]], raw if isinstance(raw, list) else raw["nodes"])
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            item_with_groups = dict(item)
            node_id = str(item.get("id", ""))
            assigned_groups = group_assignments.get(node_id)
            if assigned_groups:
                item_with_groups["parameter_groups"] = assigned_groups
            if node_id in parameter_surfaces:
                item_with_groups["parameters"] = parameter_surfaces[node_id]
            normalized_items.append(
                _enrich_manifest_parameters(
                    _merge_manifest_data(
                        item_with_groups,
                        _combined_override(
                            capabilities.get(str(item.get("id", ""))),
                            overrides.get(str(item.get("id", ""))),
                        ),
                        shared_parameter_capabilities,
                        parameter_groups,
                    ),
                    parameter_groups,
                )
            )
        manifests.extend(NodeManifest.model_validate(item) for item in normalized_items)
    extension_nodes = cast(list[dict[str, Any]], extensions.get("nodes", []))
    operation_specs = cast(list[dict[str, Any]], extensions.get("operation_specs", []))
    extension_nodes.extend(_expand_operation_spec(spec) for spec in operation_specs)
    manifests.extend(
        NodeManifest.model_validate(_enrich_manifest_parameters(item, parameter_groups))
        for item in extension_nodes
    )
    if not manifests:
        raise RuntimeError(f"No node manifests found under {RESOURCE_ROOT}")
    keys = [(manifest.id, manifest.version) for manifest in manifests]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate node manifest identity detected")
    return NodeCatalog(manifests)
