from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from domain.manifests import NodeManifest

RESOURCE_ROOT = Path(__file__).resolve().parents[2] / "resources" / "nodes"
CATALOG_EXTENSION_PATH = RESOURCE_ROOT.parent / "catalog_extensions.json"

_INITIALIZER_OPTIONS = [
    "uniform",
    "normal",
    "truncated_normal",
    "constant",
    "variance_scaling",
    "glorot_uniform",
    "glorot_normal",
    "he_uniform",
    "he_normal",
    "lecun_uniform",
    "lecun_normal",
    "zeros",
    "ones",
    "random_normal",
    "random_uniform",
    "orthogonal",
    "identity",
]
_REGULARIZER_OPTIONS = ["none", "l1", "l2", "l1_l2"]
_CONSTRAINT_OPTIONS = ["none", "max_norm", "min_max_norm", "non_neg", "unit_norm"]
_ACTIVATION_OPTIONS = [
    "linear",
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus",
    "softsign",
    "elu",
    "selu",
    "silu",
    "swish",
    "mish",
    "hard_sigmoid",
    "exponential",
]
_DTYPE_OPTIONS = [
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "bool",
]
_DATA_FORMAT_OPTIONS = ["channels_last", "channels_first"]
_PADDING_OPTIONS = ["valid", "same"]
_UNSET = object()

_NATIVE_BRANCH_ADAPTERS: dict[tuple[str, str], str] = {
    ("conv.conv1d", "pytorch"): "conv1d",
    ("conv.conv2d", "pytorch"): "conv2d",
    ("conv.max_pool2d", "pytorch"): "max_pool2d",
    ("layer.activation", "pytorch"): "activation",
    ("layer.batch_norm", "pytorch"): "batch_norm",
    ("layer.dense", "pytorch"): "dense",
    ("layer.dropout", "pytorch"): "dropout",
    ("layer.embedding", "pytorch"): "embedding",
    ("layer.flatten", "pytorch"): "flatten",
    ("layer.layer_norm", "pytorch"): "layer_norm",
    ("input.image", "pytorch"): "identity",
    ("input.tensor", "pytorch"): "identity",
    ("input.mask", "pytorch"): "identity",
    ("input.sequence", "pytorch"): "identity",
    ("input.tabular", "pytorch"): "identity",
    ("input.token", "pytorch"): "identity",
    ("merge.residual", "pytorch"): "add",
    ("merge.skip", "pytorch"): "add",
    ("output.classification", "pytorch"): "identity",
    ("output.custom", "pytorch"): "identity",
    ("output.multi_head", "pytorch"): "identity",
    ("output.regression", "pytorch"): "identity",
    ("output.segmentation", "pytorch"): "identity",
    ("output.sequence", "pytorch"): "identity",
    ("recurrent.gru", "pytorch"): "gru",
    ("recurrent.lstm", "pytorch"): "lstm",
}
_PYTORCH_ACTIVATIONS = ["linear", "relu", "gelu", "sigmoid", "tanh", "softmax"]

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
    value_schema: dict[str, Any] | None = None,
    visible_when: dict[str, Any] | None = None,
    enabled_when: dict[str, Any] | None = None,
    required_when: dict[str, Any] | None = None,
    framework_default: Any = _UNSET,
    editor_default: Any = _UNSET,
    ui: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameter = {
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
        "ui": ui or {"control": "select" if options else parameter_type},
    }
    if value_schema is not None:
        parameter["value_schema"] = value_schema
    if visible_when:
        parameter["visible_when"] = visible_when
    if enabled_when:
        parameter["enabled_when"] = enabled_when
    if required_when:
        parameter["required_when"] = required_when
    if framework_default is not _UNSET:
        parameter["framework_default"] = framework_default
    if editor_default is not _UNSET:
        parameter["editor_default"] = editor_default
    return parameter

###############################################################################
def _initializer_parameter(identifier: str, label: str, default: str) -> dict[str, Any]:
    return _parameter(
        identifier,
        label,
        "initializer",
        default,
        group="Initializers",
        options=_INITIALIZER_OPTIONS,
        ui={"control": "select_or_json", "json_type": "initializer"},
    )

###############################################################################
def _regularizer_parameter(identifier: str, label: str) -> dict[str, Any]:
    return _parameter(
        identifier,
        label,
        "regularizer",
        "none",
        group="Regularization",
        options=_REGULARIZER_OPTIONS,
        ui={"control": "select_or_json", "json_type": "regularizer"},
    )

###############################################################################
def _constraint_parameter(identifier: str, label: str) -> dict[str, Any]:
    return _parameter(
        identifier,
        label,
        "constraint",
        "none",
        group="Constraints",
        options=_CONSTRAINT_OPTIONS,
        ui={"control": "select_or_json", "json_type": "constraint"},
    )

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
    seen_group_parameters: set[str] = set()
    for reference in references:
        group = parameter_groups.get(reference)
        if group is None:
            raise RuntimeError(
                f"Unknown parameter group '{reference}' for {item.get('id', '<unknown>')}"
            )
        for parameter in group:
            identifier = str(parameter.get("id", ""))
            if identifier in seen_group_parameters:
                raise RuntimeError(
                    f"Parameter group collision for {item.get('id', '<unknown>')}: {identifier}"
                )
            seen_group_parameters.add(identifier)
            expanded.append(dict(parameter))
    explicit = item.get("parameters", [])
    if explicit is None:
        explicit = []
    if not isinstance(explicit, list):
        raise RuntimeError(f"Parameters for {item.get('id', '<unknown>')} must be a list")
    item["parameters"] = [*expanded, *cast(list[dict[str, Any]], explicit)]
    return item

###############################################################################
def _base_layer_parameters() -> list[dict[str, Any]]:
    """Safe Keras ``Layer`` constructor controls shared by layer nodes.

    ``name`` is intentionally represented by the node label. Learned weights
    are model state rather than authoring parameters and are therefore not
    exposed in the portable graph contract.
    """

    return [
        _parameter(
            "dtype",
            "Dtype",
            "dtype",
            None,
            group="Advanced",
            options=_DTYPE_OPTIONS,
            nullable=True,
            framework_default=None,
            editor_default="float32",
            ui={"control": "select", "allow_global_policy": True},
        ),
        _parameter("trainable", "Trainable", "boolean", True, group="Advanced"),
        _parameter("autocast", "Autocast", "boolean", True, group="Advanced"),
    ]

###############################################################################
def _spatial_parameters(dimensions: int) -> list[dict[str, Any]]:
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
            "padding",
            "Padding",
            "enum",
            "valid",
            group="Spatial configuration",
            options=_PADDING_OPTIONS,
        ),
        _parameter(
            "data_format",
            "Data format",
            "enum",
            "channels_last",
            group="Spatial configuration",
            options=_DATA_FORMAT_OPTIONS,
        ),
        _parameter(
            "dilation_rate",
            "Dilation rate",
            "shape",
            [1] * dimensions,
            group="Spatial configuration",
            constraints={"length": dimensions, "item_min": 1},
        ),
    ]

###############################################################################
def _pool_parameters(dimensions: int, *, include_pool_size: bool = False) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    if include_pool_size:
        parameters.append(
            _parameter(
                "pool_size",
                "Pool size",
                "shape",
                [2] * dimensions,
                group="Spatial configuration",
                constraints={"length": dimensions, "item_min": 1},
                framework_default=None,
                editor_default=[2] * dimensions,
            )
        )
    parameters.extend(
        [
            _parameter(
                "strides",
                "Strides",
                "shape",
                None,
                group="Spatial configuration",
                constraints={"length": dimensions, "item_min": 1},
                nullable=True,
            ),
            _parameter(
                "padding",
                "Padding",
                "enum",
                "valid",
                group="Spatial configuration",
                options=_PADDING_OPTIONS,
            ),
            _parameter(
                "data_format",
                "Data format",
                "enum",
                "channels_last",
                group="Spatial configuration",
                options=_DATA_FORMAT_OPTIONS,
            ),
        ]
    )
    return parameters

###############################################################################
def _safe_layer_parameters(node_id: str) -> list[dict[str, Any]]:
    """Return the safe, serializable Keras constructor surface for a node."""

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
        return [
            _parameter(
                "lora_rank",
                "LoRA rank",
                "integer",
                None,
                group="Advanced",
                constraints={"min": 1},
                nullable=True,
            ),
            _parameter(
                "lora_alpha",
                "LoRA alpha",
                "number",
                None,
                group="Advanced",
                constraints={"min": 1},
                nullable=True,
                enabled_when={"lora_rank": {"not_equals": None}},
            ),
            _parameter(
                "quantization_config",
                "Quantization config",
                "object",
                None,
                group="Advanced",
                nullable=True,
                value_schema={"type": "object"},
                ui={"control": "json"},
            ),
            *shared_kernel_controls,
        ]
    if node_id in {"conv.conv1d", "conv.conv2d", "conv.conv3d"}:
        dimensions = int(node_id[-2])
        return [
            *_spatial_parameters(dimensions),
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
    if node_id in {
        "conv.max_pool1d",
        "conv.max_pool2d",
        "conv.max_pool3d",
        "conv.average_pool1d",
        "conv.average_pool2d",
        "conv.average_pool3d",
    }:
        dimensions = int(node_id[-2])
        return _pool_parameters(dimensions)
    if node_id in {"conv.global_pool1d", "conv.global_pool2d", "conv.global_pool3d"}:
        return [
            _parameter(
                "data_format",
                "Data format",
                "enum",
                "channels_last",
                group="Spatial configuration",
                options=_DATA_FORMAT_OPTIONS,
            ),
            _parameter("keepdims", "Keep dimensions", "boolean", False, group="Shape"),
        ]
    if node_id == "layer.activation":
        return []
    if node_id == "layer.dropout":
        return [
            _parameter(
                "noise_shape",
                "Noise shape",
                "shape",
                None,
                group="Behavior",
                nullable=True,
                ui={"control": "sequence", "allow_null_items": True},
            ),
            _parameter("seed", "Seed", "integer", None, group="Behavior", nullable=True),
        ]
    if node_id == "layer.flatten":
        return [
            _parameter(
                "data_format",
                "Data format",
                "enum",
                None,
                group="Layout",
                options=_DATA_FORMAT_OPTIONS,
                nullable=True,
                framework_default=None,
                editor_default="channels_last",
            ),
        ]
    if node_id == "layer.embedding":
        return [
            _initializer_parameter("embeddings_initializer", "Embeddings initializer", "uniform"),
            _regularizer_parameter("embeddings_regularizer", "Embeddings regularizer"),
            _regularizer_parameter("activity_regularizer", "Activity regularizer"),
            _constraint_parameter("embeddings_constraint", "Embeddings constraint"),
            _parameter("mask_zero", "Mask zero", "boolean", False, group="Behavior"),
            _parameter(
                "lora_rank",
                "LoRA rank",
                "integer",
                None,
                group="Advanced",
                constraints={"min": 1},
                nullable=True,
            ),
            _parameter(
                "lora_alpha",
                "LoRA alpha",
                "number",
                None,
                group="Advanced",
                constraints={"min": 1},
                nullable=True,
                enabled_when={"lora_rank": {"not_equals": None}},
            ),
            _parameter(
                "quantization_config",
                "Quantization config",
                "object",
                None,
                group="Advanced",
                nullable=True,
                value_schema={"type": "object"},
                ui={"control": "json"},
            ),
        ]
    if node_id == "layer.batch_norm":
        return [
            _parameter("axis", "Axis", "axis", -1, group="Shape"),
            _parameter("center", "Center", "boolean", True, group="Behavior"),
            _parameter("scale", "Scale", "boolean", True, group="Behavior"),
            _initializer_parameter("beta_initializer", "Beta initializer", "zeros"),
            _initializer_parameter("gamma_initializer", "Gamma initializer", "ones"),
            _initializer_parameter("moving_mean_initializer", "Moving mean initializer", "zeros"),
            _initializer_parameter(
                "moving_variance_initializer", "Moving variance initializer", "ones"
            ),
            _regularizer_parameter("beta_regularizer", "Beta regularizer"),
            _regularizer_parameter("gamma_regularizer", "Gamma regularizer"),
            _constraint_parameter("beta_constraint", "Beta constraint"),
            _constraint_parameter("gamma_constraint", "Gamma constraint"),
            _parameter("renorm", "Renormalization", "boolean", False, group="Advanced"),
            _parameter(
                "renorm_clipping",
                "Renormalization clipping",
                "object",
                None,
                group="Advanced",
                nullable=True,
                value_schema={
                    "type": "object",
                    "properties": {
                        "rmax": {"type": "number"},
                        "rmin": {"type": "number"},
                        "dmax": {"type": "number"},
                    },
                },
                visible_when={"renorm": True},
                enabled_when={"renorm": True},
                ui={"control": "json"},
            ),
            _parameter(
                "renorm_momentum",
                "Renormalization momentum",
                "number",
                0.99,
                group="Advanced",
                constraints={"min": 0, "max": 1},
                visible_when={"renorm": True},
                enabled_when={"renorm": True},
            ),
            _parameter("synchronized", "Synchronized", "boolean", False, group="Behavior"),
        ]
    if node_id == "layer.layer_norm":
        return [
            _parameter("axis", "Axis", "list", [-1], group="Shape", constraints={"item_min": -1}),
            _parameter("center", "Center", "boolean", True, group="Behavior"),
            _parameter("scale", "Scale", "boolean", True, group="Behavior"),
            _initializer_parameter("beta_initializer", "Beta initializer", "zeros"),
            _initializer_parameter("gamma_initializer", "Gamma initializer", "ones"),
            _regularizer_parameter("beta_regularizer", "Beta regularizer"),
            _regularizer_parameter("gamma_regularizer", "Gamma regularizer"),
            _constraint_parameter("beta_constraint", "Beta constraint"),
            _constraint_parameter("gamma_constraint", "Gamma constraint"),
        ]
    if node_id == "layer.group_norm":
        return [
            _parameter("axis", "Axis", "axis", -1, group="Shape"),
            _parameter("center", "Center", "boolean", True, group="Behavior"),
            _parameter("scale", "Scale", "boolean", True, group="Behavior"),
            _initializer_parameter("beta_initializer", "Beta initializer", "zeros"),
            _initializer_parameter("gamma_initializer", "Gamma initializer", "ones"),
            _regularizer_parameter("beta_regularizer", "Beta regularizer"),
            _regularizer_parameter("gamma_regularizer", "Gamma regularizer"),
            _constraint_parameter("beta_constraint", "Beta constraint"),
            _constraint_parameter("gamma_constraint", "Gamma constraint"),
        ]
    if node_id in {"conv.transposed", "conv.depthwise", "conv.separable"}:
        dimensions = 2
        if node_id == "conv.depthwise":
            dimensions = 2
        controls: list[dict[str, Any]] = [
            _parameter(
                "dimensions",
                "Dimensions",
                "enum",
                dimensions,
                group="Spatial configuration",
                options=[1, 2, 3] if node_id == "conv.transposed" else [1, 2],
            ),
            _parameter(
                "strides",
                "Strides",
                "shape",
                [1] * dimensions,
                group="Spatial configuration",
                constraints={"item_min": 1},
            ),
            _parameter(
                "padding",
                "Padding",
                "enum",
                "valid",
                group="Spatial configuration",
                options=_PADDING_OPTIONS,
            ),
            _parameter(
                "data_format",
                "Data format",
                "enum",
                "channels_last",
                group="Spatial configuration",
                options=_DATA_FORMAT_OPTIONS,
            ),
            _parameter(
                "dilation_rate",
                "Dilation rate",
                "shape",
                [1] * dimensions,
                group="Spatial configuration",
                constraints={"item_min": 1},
            ),
            _parameter(
                "activation",
                "Activation",
                "activation",
                "linear",
                group="Activation",
                options=_ACTIVATION_OPTIONS,
            ),
            _parameter("use_bias", "Use bias", "boolean", True, group="General"),
        ]
        if node_id == "conv.transposed":
            controls.insert(
                1,
                _parameter(
                    "output_padding",
                    "Output padding",
                    "shape",
                    None,
                    group="Spatial configuration",
                    constraints={"item_min": 0},
                    nullable=True,
                ),
            )
            controls.extend(shared_kernel_controls)
        elif node_id == "conv.depthwise":
            controls[0] = _parameter(
                "dimensions",
                "Dimensions",
                "enum",
                2,
                group="Spatial configuration",
                options=[1, 2],
            )
            controls.insert(
                1,
                _parameter(
                    "depth_multiplier",
                    "Depth multiplier",
                    "integer",
                    1,
                    group="Channels",
                    constraints={"min": 1},
                ),
            )
            controls.extend(
                [
                    _initializer_parameter("depthwise_initializer", "Depthwise initializer", "glorot_uniform"),
                    _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
                    _regularizer_parameter("depthwise_regularizer", "Depthwise regularizer"),
                    _regularizer_parameter("bias_regularizer", "Bias regularizer"),
                    _regularizer_parameter("activity_regularizer", "Activity regularizer"),
                    _constraint_parameter("depthwise_constraint", "Depthwise constraint"),
                    _constraint_parameter("bias_constraint", "Bias constraint"),
                ]
            )
        else:
            controls.extend(
                [
                    _parameter(
                        "depth_multiplier",
                        "Depth multiplier",
                        "integer",
                        1,
                        group="Channels",
                        constraints={"min": 1},
                    ),
                    _initializer_parameter("depthwise_initializer", "Depthwise initializer", "glorot_uniform"),
                    _initializer_parameter("pointwise_initializer", "Pointwise initializer", "glorot_uniform"),
                    _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
                    _regularizer_parameter("depthwise_regularizer", "Depthwise regularizer"),
                    _regularizer_parameter("pointwise_regularizer", "Pointwise regularizer"),
                    _regularizer_parameter("bias_regularizer", "Bias regularizer"),
                    _regularizer_parameter("activity_regularizer", "Activity regularizer"),
                    _constraint_parameter("depthwise_constraint", "Depthwise constraint"),
                    _constraint_parameter("pointwise_constraint", "Pointwise constraint"),
                    _constraint_parameter("bias_constraint", "Bias constraint"),
                ]
            )
        return controls
    if node_id == "conv.upsampling":
        return [
            _parameter("dimensions", "Dimensions", "enum", 2, group="Spatial configuration", options=[1, 2, 3]),
            _parameter("data_format", "Data format", "enum", "channels_last", group="Spatial configuration", options=_DATA_FORMAT_OPTIONS),
            _parameter("interpolation", "Interpolation", "enum", "nearest", group="Spatial configuration", options=["nearest", "bilinear"], visible_when={"dimensions": 2}, enabled_when={"dimensions": 2}),
        ]
    if node_id in {"conv.padding", "conv.cropping"}:
        return [
            _parameter("dimensions", "Dimensions", "enum", 2, group="Spatial configuration", options=[1, 2, 3]),
            _parameter("data_format", "Data format", "enum", "channels_last", group="Spatial configuration", options=_DATA_FORMAT_OPTIONS),
        ]
    if node_id == "recurrent.simple_rnn":
        return [
            _parameter("activation", "Activation", "activation", "tanh", group="Activation", options=_ACTIVATION_OPTIONS),
            _parameter("use_bias", "Use bias", "boolean", True, group="General"),
            _initializer_parameter("kernel_initializer", "Kernel initializer", "glorot_uniform"),
            _initializer_parameter("recurrent_initializer", "Recurrent initializer", "orthogonal"),
            _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
            _regularizer_parameter("kernel_regularizer", "Kernel regularizer"),
            _regularizer_parameter("recurrent_regularizer", "Recurrent regularizer"),
            _regularizer_parameter("bias_regularizer", "Bias regularizer"),
            _regularizer_parameter("activity_regularizer", "Activity regularizer"),
            _constraint_parameter("kernel_constraint", "Kernel constraint"),
            _constraint_parameter("recurrent_constraint", "Recurrent constraint"),
            _constraint_parameter("bias_constraint", "Bias constraint"),
            _parameter("dropout", "Dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("recurrent_dropout", "Recurrent dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("return_state", "Return state", "boolean", False, group="Behavior"),
            _parameter("go_backwards", "Go backwards", "boolean", False, group="Behavior"),
            _parameter("stateful", "Stateful", "boolean", False, group="Behavior"),
            _parameter("unroll", "Unroll", "boolean", False, group="Behavior"),
            _parameter("seed", "Seed", "integer", None, group="Behavior", nullable=True),
        ]
    if node_id in {"recurrent.gru", "recurrent.lstm"}:
        recurrent_controls = [
            _parameter("activation", "Activation", "activation", "tanh", group="Activation", options=_ACTIVATION_OPTIONS),
            _parameter("recurrent_activation", "Recurrent activation", "activation", "sigmoid", group="Activation", options=_ACTIVATION_OPTIONS),
            _parameter("use_bias", "Use bias", "boolean", True, group="General"),
            _initializer_parameter("kernel_initializer", "Kernel initializer", "glorot_uniform"),
            _initializer_parameter("recurrent_initializer", "Recurrent initializer", "orthogonal"),
            _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
            _regularizer_parameter("kernel_regularizer", "Kernel regularizer"),
            _regularizer_parameter("recurrent_regularizer", "Recurrent regularizer"),
            _regularizer_parameter("bias_regularizer", "Bias regularizer"),
            _regularizer_parameter("activity_regularizer", "Activity regularizer"),
            _constraint_parameter("kernel_constraint", "Kernel constraint"),
            _constraint_parameter("recurrent_constraint", "Recurrent constraint"),
            _constraint_parameter("bias_constraint", "Bias constraint"),
            _parameter("dropout", "Dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("recurrent_dropout", "Recurrent dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("return_state", "Return state", "boolean", False, group="Behavior"),
            _parameter("go_backwards", "Go backwards", "boolean", False, group="Behavior"),
            _parameter("stateful", "Stateful", "boolean", False, group="Behavior"),
            _parameter("unroll", "Unroll", "boolean", False, group="Behavior"),
            _parameter("seed", "Seed", "integer", None, group="Behavior", nullable=True),
            _parameter(
                "use_cudnn",
                "Use cuDNN",
                "enum",
                "auto",
                group="Backend-specific",
                options=["auto", True, False],
            ),
        ]
        recurrent_controls.append(
            _parameter(
                "reset_after" if node_id == "recurrent.gru" else "unit_forget_bias",
                "Reset after" if node_id == "recurrent.gru" else "Unit forget bias",
                "boolean",
                True,
                group="Behavior",
            )
        )
        return recurrent_controls
    if node_id in {"recurrent.bidirectional", "recurrent.stack"}:
        return [
            _parameter("merge_mode", "Merge mode", "enum", "concat", group="Behavior", options=["sum", "mul", "concat", "ave"], nullable=True),
            _parameter("activation", "Activation", "activation", "tanh", group="Activation", options=_ACTIVATION_OPTIONS),
            _parameter("recurrent_activation", "Recurrent activation", "activation", "sigmoid", group="Activation", options=_ACTIVATION_OPTIONS),
            _parameter("use_bias", "Use bias", "boolean", True, group="General"),
            _initializer_parameter("kernel_initializer", "Kernel initializer", "glorot_uniform"),
            _initializer_parameter("recurrent_initializer", "Recurrent initializer", "orthogonal"),
            _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
            _regularizer_parameter("kernel_regularizer", "Kernel regularizer"),
            _regularizer_parameter("recurrent_regularizer", "Recurrent regularizer"),
            _regularizer_parameter("bias_regularizer", "Bias regularizer"),
            _regularizer_parameter("activity_regularizer", "Activity regularizer"),
            _constraint_parameter("kernel_constraint", "Kernel constraint"),
            _constraint_parameter("recurrent_constraint", "Recurrent constraint"),
            _constraint_parameter("bias_constraint", "Bias constraint"),
            _parameter("dropout", "Dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("recurrent_dropout", "Recurrent dropout", "number", 0.0, group="Regularization", constraints={"min": 0, "max": 0.999}),
            _parameter("go_backwards", "Go backwards", "boolean", False, group="Behavior"),
            _parameter("stateful", "Stateful", "boolean", False, group="Behavior"),
            _parameter("unroll", "Unroll", "boolean", False, group="Behavior"),
            _parameter("seed", "Seed", "integer", None, group="Behavior", nullable=True),
        ]
    if node_id in {"attention.multi_head", "attention.self", "attention.cross"}:
        return [
            _parameter("value_dim", "Value dim", "integer", None, group="Attention", constraints={"min": 1}, nullable=True),
            _parameter("output_shape", "Output shape", "shape", None, group="Attention", nullable=True),
            _parameter("attention_axes", "Attention axes", "list", None, group="Attention", nullable=True),
            _parameter("sliding_window", "Sliding window", "integer", None, group="Attention", constraints={"min": 1}, nullable=True),
            _parameter("flash_attention", "Flash attention", "boolean", None, group="Backend-specific", nullable=True),
            _parameter("use_bias", "Use bias", "boolean", True, group="General"),
            _parameter("use_gate", "Use gate", "boolean", False, group="Advanced"),
            _initializer_parameter("kernel_initializer", "Kernel initializer", "glorot_uniform"),
            _initializer_parameter("bias_initializer", "Bias initializer", "zeros"),
            _regularizer_parameter("kernel_regularizer", "Kernel regularizer"),
            _regularizer_parameter("bias_regularizer", "Bias regularizer"),
            _regularizer_parameter("activity_regularizer", "Activity regularizer"),
            _constraint_parameter("kernel_constraint", "Kernel constraint"),
            _constraint_parameter("bias_constraint", "Bias constraint"),
            _parameter("seed", "Seed", "integer", None, group="Behavior", nullable=True),
        ]
    if node_id in {
        "transformer.encoder_block",
        "transformer.stack",
        "transformer.decoder_block",
    }:
        return [
            _parameter("num_heads", "Num heads", "integer", 8, group="Attention", constraints={"min": 1}),
            _parameter("key_dim", "Key dimension", "integer", 64, group="Attention", constraints={"min": 1}),
            _parameter(
                "dropout",
                "Dropout",
                "number",
                0.0,
                group="Regularization",
                constraints={"min": 0, "max": 0.999},
            ),
            _parameter("layers", "Layers", "integer", 1, group="Structure", constraints={"min": 1}),
            _parameter(
                "activation",
                "Feed-forward activation",
                "activation",
                "gelu",
                group="Activation",
                options=_ACTIVATION_OPTIONS,
            ),
            _parameter(
                "ffn_multiplier",
                "Feed-forward multiplier",
                "integer",
                4,
                group="Shape",
                constraints={"min": 1},
            ),
            _parameter("pre_norm", "Pre-normalization", "boolean", True, group="Behavior"),
        ]
    if node_id == "transformer.feed_forward":
        return [
            _parameter(
                "activation",
                "Feed-forward activation",
                "activation",
                "gelu",
                group="Activation",
                options=_ACTIVATION_OPTIONS,
            ),
            _parameter(
                "ffn_multiplier",
                "Feed-forward multiplier",
                "integer",
                4,
                group="Shape",
                constraints={"min": 1},
            ),
            _parameter(
                "dropout",
                "Dropout",
                "number",
                0.0,
                group="Regularization",
                constraints={"min": 0, "max": 0.999},
            ),
        ]
    return []

###############################################################################
def _enrich_manifest_parameters(
    raw: dict[str, Any],
    parameter_groups: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Normalize the catalog into the complete safe inspector contract."""

    item = _expand_parameter_groups(raw, parameter_groups or {})
    node_id = str(item.get("id", ""))
    parameters: list[dict[str, Any]] = []
    extra_parameters = _safe_layer_parameters(node_id)
    if node_id.startswith(("layer.", "conv.", "recurrent.", "attention.", "transformer.")):
        extra_parameters = [*_base_layer_parameters(), *extra_parameters]
    if node_id in {
        "transformer.encoder_block",
        "transformer.stack",
        "transformer.decoder_block",
        "transformer.feed_forward",
    }:
        # The JSON resources intentionally share a compact placeholder shape for
        # transformer blocks.  The executable catalog owns the node contract and
        # replaces that placeholder with the parameters each implementation uses.
        item["parameters"] = []
    for parameter in [*item.get("parameters", []), *extra_parameters]:
        normalized = dict(parameter)
        identifier = str(normalized.get("id", ""))
        if identifier in {"activation", "recurrent_activation"}:
            normalized["type"] = "activation"
            normalized["options"] = list(_ACTIVATION_OPTIONS)
        elif identifier == "dtype":
            normalized["type"] = "dtype"
            normalized["options"] = list(_DTYPE_OPTIONS)
        if not str(normalized.get("description", "")).strip():
            normalized["description"] = (
                f"Configure {normalized.get('display_name', normalized.get('id', 'this value'))}."
            )
        if not str(normalized.get("group", "")).strip() or normalized.get("group") == "General":
            if "initializer" in identifier:
                normalized["group"] = "Initializers"
            elif "regularizer" in identifier:
                normalized["group"] = "Regularization"
            elif "constraint" in identifier:
                normalized["group"] = "Constraints"
            elif identifier in {
                "dropout",
                "recurrent_dropout",
                "seed",
                "return_sequences",
                "return_state",
                "go_backwards",
                "stateful",
                "unroll",
                "use_bias",
                "center",
                "scale",
                "mask_zero",
                "synchronized",
                "reset_after",
                "unit_forget_bias",
                "pre_norm",
            }:
                normalized["group"] = "Behavior"
            elif identifier in {
                "shape",
                "kernel_size",
                "strides",
                "dilation_rate",
                "pool_size",
                "axis",
                "groups",
                "filters",
                "units",
                "vocabulary_size",
                "embedding_dim",
                "num_heads",
                "key_dim",
                "value_dim",
                "output_shape",
                "attention_axes",
                "ffn_multiplier",
            }:
                normalized["group"] = "Shape"
            else:
                normalized["group"] = "General"
        normalized.setdefault("nullable", normalized.get("default") is None)
        normalized.setdefault("ui", {"control": normalized.get("type", "text")})
        normalized.setdefault("framework_default", normalized.get("default"))
        normalized.setdefault("editor_default", normalized.get("default"))
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
    default_implementation_adapters: dict[str, str] | None = None,
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
        if (
            mapping.get("supported", True)
            and not mapping.get("symbol")
            and not mapping.get("adapter")
        ):
            mapping["adapter"] = adapter
        implementations[implementation] = mapping
    node_id = str(item.get("id", ""))
    for (adapter_node_id, implementation), adapter in _NATIVE_BRANCH_ADAPTERS.items():
        if adapter_node_id != node_id:
            continue
        mapping = dict(implementations.get(implementation, {}))
        if mapping.get("supported", True) and not mapping.get("symbol") and not mapping.get("adapter"):
            mapping["adapter"] = adapter
        if implementation == "pytorch" and node_id in {
            "conv.conv1d",
            "conv.conv2d",
            "layer.activation",
            "layer.dense",
        }:
            mapping_parameters = dict(mapping.get("parameters", {}))
            mapping_parameters.setdefault(
                "activation",
                {
                    "allowed_values": _PYTORCH_ACTIVATIONS,
                    "notes": "The native PyTorch adapter supports the listed activation functions.",
                },
            )
            mapping["parameters"] = mapping_parameters
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
    parameter_groups, group_assignments = _parameter_group_catalog(extensions)
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
        items = cast(list[dict[str, Any]], raw if isinstance(raw, list) else raw["nodes"])
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            item_with_groups = dict(item)
            assigned_groups = group_assignments.get(str(item.get("id", "")))
            if assigned_groups:
                item_with_groups["parameter_groups"] = assigned_groups
            normalized_items.append(
                _enrich_manifest_parameters(
                    _merge_manifest_data(
                        item_with_groups,
                        _combined_override(
                            capabilities.get(str(item.get("id", ""))),
                            overrides.get(str(item.get("id", ""))),
                        ),
                        shared_parameter_capabilities,
                        default_implementation_adapters,
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
