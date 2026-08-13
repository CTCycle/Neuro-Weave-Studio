from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from domain.manifests import NodeManifest

RESOURCE_ROOT = Path(__file__).resolve().parents[2] / "resources" / "nodes"

_PORTABLE_CORE = {
    "input.image", "input.mask", "input.sequence", "input.tabular", "input.tensor", "input.token",
    "layer.activation", "layer.batch_norm", "layer.dense", "layer.dropout", "layer.embedding",
    "layer.flatten", "layer.layer_norm", "conv.conv2d", "conv.max_pool2d", "recurrent.gru",
    "recurrent.lstm", "merge.add", "merge.concatenate", "merge.residual", "merge.skip",
    "output.classification", "output.custom", "output.multi_head", "output.regression",
    "output.segmentation", "output.sequence",
}

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
    if node_id == "conv.conv2d":
        return [
            _parameter(
                "strides",
                "Strides",
                "shape",
                [1, 1],
                group="Spatial configuration",
                constraints={"length": 2, "item_min": 1},
            ),
            _parameter(
                "dilation_rate",
                "Dilation rate",
                "shape",
                [1, 1],
                group="Spatial configuration",
                constraints={"length": 2, "item_min": 1},
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
def _normalize_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the checked-in catalog into the v2 four-target contract."""

    item = dict(raw)
    original = dict(item.get("frameworks", {}))
    mappings: dict[str, Any] = {}
    for key, value in original.items():
        target = key
        if target is None:
            continue
        mappings[target] = value
    node_id = str(item.get("id", ""))
    keras_mapping = mappings.get("keras", {"supported": False, "notes": "Design-only node"})
    mappings.setdefault("keras", keras_mapping)
    mappings.setdefault("tensorflow", mappings.get("tensorflow", {"supported": False, "notes": "Target-limited"}))
    mappings.setdefault("pytorch", mappings.get("pytorch", {"supported": False, "notes": "Target-limited"}))
    if node_id in _PORTABLE_CORE and bool(keras_mapping.get("supported", False)):
        mappings["tensorflow"] = {"supported": True, "symbol": mappings["tensorflow"].get("symbol")}
        mappings["pytorch"] = {"supported": True, "symbol": mappings["pytorch"].get("symbol")}
        mappings["jax"] = {"supported": True, "symbol": "flax.linen"}
    else:
        mappings.setdefault("jax", {"supported": False, "notes": "Target-limited or design-only node"})
    item["frameworks"] = mappings
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
    manifests: list[NodeManifest] = []
    for path in sorted(RESOURCE_ROOT.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = cast(
            list[dict[str, Any]], raw if isinstance(raw, list) else raw["nodes"]
        )
        manifests.extend(NodeManifest.model_validate(_normalize_manifest(item)) for item in items)
    if not manifests:
        raise RuntimeError(f"No node manifests found under {RESOURCE_ROOT}")
    keys = [(manifest.id, manifest.version) for manifest in manifests]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate node manifest identity detected")
    return NodeCatalog(manifests)
