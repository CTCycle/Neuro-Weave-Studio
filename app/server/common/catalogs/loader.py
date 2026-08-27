from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from domain.manifests import NodeManifest

RESOURCE_ROOT = Path(__file__).resolve().parents[2] / "resources" / "nodes"
CATALOG_EXTENSION_PATH = RESOURCE_ROOT.parent / "catalog_extensions.json"

_INITIALIZER_OPTIONS = [
    "uniform",
    "glorot_uniform",
    "glorot_normal",
    "he_uniform",
    "he_normal",
    "zeros",
    "ones",
    "random_normal",
    "random_uniform",
    "orthogonal",
]
_REGULARIZER_OPTIONS = ["none", "l1", "l2", "l1_l2"]
_CONSTRAINT_OPTIONS = ["none", "max_norm", "min_max_norm", "non_neg", "unit_norm"]

###############################################################################
def _parameter(
    identifier: str,
    display_name: str,
    parameter_type: str,
    default: Any,
    *,
    group: str,
    options: list[Any] | None = None,
    constraints: dict[str, Any] | None = None,
    nullable: bool = False,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "display_name": display_name,
        "type": parameter_type,
        "default": default,
        "required": False,
        "options": options or [],
        "constraints": constraints or {},
        "description": f"Configure the {display_name.lower()} used by this layer.",
        "group": group,
        "nullable": nullable,
        "ui": {"control": "select" if options else parameter_type},
    }

###############################################################################
def _initializer_parameter(identifier: str, label: str, default: str) -> dict[str, Any]:
    return _parameter(
        identifier,
        label,
        "enum",
        default,
        group="Initializers",
        options=_INITIALIZER_OPTIONS,
    )

###############################################################################
def _regularizer_parameter(identifier: str, label: str) -> dict[str, Any]:
    return _parameter(
        identifier,
        label,
        "enum",
        "none",
        group="Regularization",
        options=_REGULARIZER_OPTIONS,
    )

###############################################################################
def _constraint_parameter(identifier: str, label: str) -> dict[str, Any]:
    return _parameter(
        identifier,
        label,
        "enum",
        "none",
        group="Constraints",
        options=_CONSTRAINT_OPTIONS,
    )

###############################################################################
def _safe_layer_parameters(node_id: str) -> list[dict[str, Any]]:
    """Return the serializable Keras constructor controls for key layers.

    The checked-in catalog intentionally stays compact and human-editable. This
    runtime contract adds the safe string/number/boolean controls that map to
    Keras' public constructors without exposing callables or arbitrary objects.
    """

    shared_kernel_controls = [
        _initializer_parameter("kernel_initializer", "Kernel initializer", "glorot_uniform"),
        _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
        _regularizer_parameter("kernel_regularizer", "Kernel regularizer"),
        _regularizer_parameter("bias_regularizer", "Bias regularizer"),
        _regularizer_parameter("activity_regularizer", "Activity regularizer"),
        _constraint_parameter("kernel_constraint", "Kernel constraint"),
        _constraint_parameter("bias_constraint", "Bias constraint"),
    ]
    if node_id == "layer.dense":
        return shared_kernel_controls
    if node_id in {"conv.conv1d", "conv.conv2d"}:
        dimensions = 1 if node_id == "conv.conv1d" else 2
        return [
            _parameter(
                "strides",
                "Strides",
                "shape",
                [1] * dimensions,
                group="Spatial configuration",
                constraints={"length": dimensions, "item_min": 1},
            ),
            _parameter(
                "dilation_rate",
                "Dilation rate",
                "shape",
                [1] * dimensions,
                group="Spatial configuration",
                constraints={"length": dimensions, "item_min": 1},
            ),
            _parameter(
                "groups",
                "Groups",
                "integer",
                1,
                group="Spatial configuration",
                constraints={"min": 1},
            ),
            *shared_kernel_controls,
        ]
    if node_id == "conv.max_pool2d":
        return [
            _parameter(
                "strides",
                "Strides",
                "shape",
                None,
                group="Spatial configuration",
                constraints={"length": 2, "item_min": 1},
                nullable=True,
            ),
            _parameter(
                "padding",
                "Padding",
                "enum",
                "valid",
                group="Spatial configuration",
                options=["valid", "same"],
            ),
            _parameter(
                "data_format",
                "Data format",
                "enum",
                "channels_last",
                group="Spatial configuration",
                options=["channels_last", "channels_first"],
            ),
        ]
    if node_id == "layer.dropout":
        return [
            _parameter("seed", "Seed", "integer", None, group="Behavior", nullable=True),
        ]
    if node_id == "layer.flatten":
        return [
            _parameter(
                "data_format",
                "Data format",
                "enum",
                "channels_last",
                group="Layout",
                options=["channels_last", "channels_first"],
            ),
        ]
    if node_id == "layer.embedding":
        return [
            _initializer_parameter("embeddings_initializer", "Embeddings initializer", "uniform"),
            _regularizer_parameter("embeddings_regularizer", "Embeddings regularizer"),
            _regularizer_parameter("activity_regularizer", "Activity regularizer"),
            _constraint_parameter("embeddings_constraint", "Embeddings constraint"),
            _parameter("mask_zero", "Mask zero", "boolean", False, group="Behavior"),
            _parameter("lora_rank", "LoRA rank", "integer", None, group="Behavior", constraints={"min": 1}, nullable=True),
        ]
    if node_id == "layer.batch_norm":
        return [
            _parameter("axis", "Axis", "axis", -1, group="Shape"),
            _parameter("center", "Center", "boolean", True, group="Behavior"),
            _parameter("scale", "Scale", "boolean", True, group="Behavior"),
            _initializer_parameter("beta_initializer", "Beta initializer", "zeros"),
            _initializer_parameter("gamma_initializer", "Gamma initializer", "ones"),
            _initializer_parameter("moving_mean_initializer", "Moving mean initializer", "zeros"),
            _initializer_parameter("moving_variance_initializer", "Moving variance initializer", "ones"),
            _regularizer_parameter("beta_regularizer", "Beta regularizer"),
            _regularizer_parameter("gamma_regularizer", "Gamma regularizer"),
            _constraint_parameter("beta_constraint", "Beta constraint"),
            _constraint_parameter("gamma_constraint", "Gamma constraint"),
            _parameter("synchronized", "Synchronized", "boolean", False, group="Behavior"),
        ]
    if node_id == "layer.layer_norm":
        return [
            _parameter("axis", "Axis", "list", [-1], group="Shape", constraints={"length": 1}),
            _parameter("center", "Center", "boolean", True, group="Behavior"),
            _parameter("scale", "Scale", "boolean", True, group="Behavior"),
            _initializer_parameter("beta_initializer", "Beta initializer", "zeros"),
            _initializer_parameter("gamma_initializer", "Gamma initializer", "ones"),
            _regularizer_parameter("beta_regularizer", "Beta regularizer"),
            _regularizer_parameter("gamma_regularizer", "Gamma regularizer"),
            _constraint_parameter("beta_constraint", "Beta constraint"),
            _constraint_parameter("gamma_constraint", "Gamma constraint"),
        ]
    if node_id in {"recurrent.gru", "recurrent.lstm"}:
        recurrent_controls = [
            _parameter("activation", "Activation", "enum", "tanh", group="Behavior", options=["tanh", "relu"]),
            _parameter("recurrent_activation", "Recurrent activation", "enum", "sigmoid", group="Behavior", options=["sigmoid", "hard_sigmoid"]),
            _parameter("use_bias", "Use bias", "boolean", True, group="Behavior"),
            _initializer_parameter("kernel_initializer", "Kernel initializer", "glorot_uniform"),
            _initializer_parameter("recurrent_initializer", "Recurrent initializer", "orthogonal"),
            _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
            _parameter("dropout", "Dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("recurrent_dropout", "Recurrent dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("return_state", "Return state", "boolean", False, group="Behavior"),
            _parameter("go_backwards", "Go backwards", "boolean", False, group="Behavior"),
            _parameter("stateful", "Stateful", "boolean", False, group="Behavior"),
            _parameter("unroll", "Unroll", "boolean", False, group="Behavior"),
            _parameter("seed", "Seed", "integer", None, group="Behavior", nullable=True),
        ]
        if node_id == "recurrent.gru":
            recurrent_controls.append(_parameter("reset_after", "Reset after", "boolean", True, group="Behavior"))
        else:
            recurrent_controls.append(_parameter("unit_forget_bias", "Unit forget bias", "boolean", True, group="Behavior"))
        return recurrent_controls
    if node_id in {"attention.multi_head", "attention.self"}:
        return [
            _parameter("value_dim", "Value dim", "integer", None, group="Attention", constraints={"min": 1}, nullable=True),
            _parameter("output_shape", "Output shape", "shape", None, group="Attention", nullable=True),
            _parameter("attention_axes", "Attention axes", "list", None, group="Attention", nullable=True),
            _parameter("use_bias", "Use bias", "boolean", True, group="Behavior"),
            _initializer_parameter("kernel_initializer", "Kernel initializer", "glorot_uniform"),
            _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
        ]
    if node_id == "transformer.encoder_block":
        return [
            _parameter("activation", "Feed-forward activation", "enum", "gelu", group="Behavior", options=["gelu", "relu"]),
            _parameter("ffn_multiplier", "Feed-forward multiplier", "integer", 4, group="Shape", constraints={"min": 1}),
            _parameter("pre_norm", "Pre-normalization", "boolean", True, group="Behavior"),
        ]
    return []

###############################################################################
def _enrich_manifest_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    """Add the safe constructor controls used by the inspector."""

    item = dict(raw)
    node_id = str(item.get("id", ""))
    parameters: list[dict[str, Any]] = []
    for parameter in [*item.get("parameters", []), *_safe_layer_parameters(node_id)]:
        normalized = dict(parameter)
        identifier = str(normalized.get("id", ""))
        if not str(normalized.get("description", "")).strip():
            normalized["description"] = f"Configure {normalized.get('display_name', normalized.get('id', 'this value'))}."
        if not str(normalized.get("group", "")).strip() or normalized.get("group") == "General":
            if "initializer" in identifier:
                normalized["group"] = "Initializers"
            elif "regularizer" in identifier:
                normalized["group"] = "Regularization"
            elif "constraint" in identifier:
                normalized["group"] = "Constraints"
            elif identifier in {"dropout", "recurrent_dropout", "seed", "return_sequences", "return_state", "go_backwards", "stateful", "unroll", "use_bias", "center", "scale", "mask_zero", "synchronized", "reset_after", "unit_forget_bias", "pre_norm"}:
                normalized["group"] = "Behavior"
            elif identifier in {"shape", "kernel_size", "strides", "dilation_rate", "pool_size", "axis", "groups", "filters", "units", "vocabulary_size", "embedding_dim", "num_heads", "key_dim", "value_dim", "output_shape", "attention_axes", "ffn_multiplier"}:
                normalized["group"] = "Shape"
            else:
                normalized["group"] = "General"
        normalized.setdefault("nullable", normalized.get("default") is None)
        normalized.setdefault("ui", {"control": normalized.get("type", "text")})
        parameters.append(normalized)
    unique_parameters: list[dict[str, Any]] = []
    seen_parameters: set[str] = set()
    for parameter in parameters:
        identifier = str(parameter.get("id", ""))
        if identifier in seen_parameters:
            continue
        seen_parameters.add(identifier)
        unique_parameters.append(parameter)
    item["parameters"] = unique_parameters
    return item

###############################################################################
def _merge_manifest_data(
    raw: dict[str, Any],
    override: dict[str, Any] | None,
    shared_parameter_capabilities: dict[str, dict[str, Any]] | None = None,
    default_implementation_adapters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply checked-in declarative capability/operation data to a node."""

    item = dict(raw)
    raw_parameters = raw.get("parameters", [])
    parameter_items = [*cast(list[Any], raw_parameters)] if isinstance(raw_parameters, list) else []
    parameter_items.extend(_safe_layer_parameters(str(raw.get("id", ""))))
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
    for implementation, adapter in (default_implementation_adapters or {}).items():
        mapping = dict(implementations.get(implementation, {}))
        if mapping.get("supported", True) and not mapping.get("symbol") and not mapping.get("adapter"):
            mapping["adapter"] = adapter
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
        "outputs": [{
            "id": "output",
            "display_name": "Output",
            "kind": "tensor",
            "required": True,
            "accepts_multiple": False,
            "rank": None,
        }],
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
    overrides = cast(dict[str, dict[str, Any]], extensions.get("overrides", {}))
    capabilities = cast(dict[str, dict[str, Any]], extensions.get("capabilities", {}))
    shared_parameter_capabilities = cast(
        dict[str, dict[str, Any]], extensions.get("shared_parameter_capabilities", {})
    )
    default_implementation_adapters = cast(
        dict[str, str], extensions.get("default_implementation_adapters", {})
    )
    manifests: list[NodeManifest] = []
    for path in sorted(RESOURCE_ROOT.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = cast(
            list[dict[str, Any]], raw if isinstance(raw, list) else raw["nodes"]
        )
        manifests.extend(
            NodeManifest.model_validate(
                _enrich_manifest_parameters(
                    _merge_manifest_data(
                        item,
                        _combined_override(
                            capabilities.get(str(item.get("id", ""))),
                            overrides.get(str(item.get("id", ""))),
                        ),
                        shared_parameter_capabilities,
                        default_implementation_adapters,
                    )
                )
            )
            for item in items
        )
    extension_nodes = cast(list[dict[str, Any]], extensions.get("nodes", []))
    operation_specs = cast(list[dict[str, Any]], extensions.get("operation_specs", []))
    extension_nodes.extend(_expand_operation_spec(spec) for spec in operation_specs)
    manifests.extend(
        NodeManifest.model_validate(_enrich_manifest_parameters(item))
        for item in extension_nodes
    )
    if not manifests:
        raise RuntimeError(f"No node manifests found under {RESOURCE_ROOT}")
    keys = [(manifest.id, manifest.version) for manifest in manifests]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate node manifest identity detected")
    return NodeCatalog(manifests)
