from __future__ import annotations

from domain.enums import CodeTarget

PORTABLE_CORE = frozenset(
    {
        "input.image", "input.mask", "input.sequence", "input.tabular", "input.tensor", "input.token",
        "layer.activation", "layer.batch_norm", "layer.dense", "layer.dropout", "layer.embedding",
        "layer.flatten", "layer.layer_norm", "conv.conv2d", "conv.max_pool2d", "recurrent.gru",
        "recurrent.lstm", "merge.add", "merge.concatenate", "merge.residual", "merge.skip",
        "output.classification", "output.custom", "output.multi_head", "output.regression",
        "output.segmentation", "output.sequence",
    }
)
KERAS_EXTENSIONS = frozenset(
    {"attention.multi_head", "attention.self", "transformer.encoder_block", "transformer.positional_encoding"}
)

###############################################################################
def adapter_supported(code_target: CodeTarget, definition_id: str) -> bool:
    if code_target is CodeTarget.KERAS:
        return definition_id in PORTABLE_CORE or definition_id in KERAS_EXTENSIONS
    return definition_id in PORTABLE_CORE

###############################################################################
def adapter_name(code_target: CodeTarget, definition_id: str) -> str:
    return f"{code_target.value}:{definition_id}"
