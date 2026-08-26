from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from zipfile import BadZipFile, ZipFile

from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position, ProjectCreate

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_CONFIG_BYTES = 5 * 1024 * 1024
_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

SUPPORTED_LAYERS = {
    "InputLayer",
    "Dense",
    "Activation",
    "Dropout",
    "Flatten",
    "BatchNormalization",
    "LayerNormalization",
    "Embedding",
    "Conv2D",
    "MaxPooling2D",
    "LSTM",
    "GRU",
    "MultiHeadAttention",
    "Add",
    "Concatenate",
}

###############################################################################
class KerasArchitectureImportError(ValueError):

    # -------------------------------------------------------------------------
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

###############################################################################
@dataclass(frozen=True)
class ParsedKerasArchitecture:
    payload: ProjectCreate
    warnings: list[str]

###############################################################################
def parse_keras_architecture(
    payload: bytes,
    *,
    execution_backend: ExecutionBackend,
    name_override: str | None = None,
) -> ParsedKerasArchitecture:
    if not payload:
        raise KerasArchitectureImportError("empty_archive", "The uploaded Keras archive is empty")
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise KerasArchitectureImportError("archive_too_large", "The Keras archive exceeds the 50 MB import limit")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            names = set(archive.namelist())
            if "config.json" not in names:
                raise KerasArchitectureImportError("missing_config", "The .keras archive does not contain config.json")
            config_info = archive.getinfo("config.json")
            if config_info.file_size > MAX_CONFIG_BYTES:
                raise KerasArchitectureImportError("config_too_large", "The serialized Keras configuration exceeds the import limit")
            raw_config = archive.read("config.json")
    except BadZipFile as exc:
        raise KerasArchitectureImportError("invalid_archive", "The uploaded file is not a valid .keras ZIP archive") from exc
    try:
        document = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise KerasArchitectureImportError("invalid_config", "The Keras configuration is not valid JSON") from exc
    graph, model_name = _graph_from_config(document)
    return ParsedKerasArchitecture(
        payload=ProjectCreate(
            name=name_override or model_name or "Imported Keras architecture",
            description="Architecture imported from a .keras archive. Learned weights are not imported.",
            execution_backend=execution_backend,
            keras_enabled=True,
            graph=graph,
        ),
        warnings=[
            "The editable graph contains architecture only; trained weights from the .keras archive were not imported.",
            f"The imported Keras architecture is associated with the selected {execution_backend.value} computational backend.",
        ],
    )


###############################################################################
def _graph_from_config(document: Any) -> tuple[ModelGraph, str]:
    if not isinstance(document, dict):
        raise KerasArchitectureImportError("invalid_config", "Keras config.json must contain an object")
    class_name = document.get("class_name")
    config = document.get("config")
    if class_name not in {"Functional", "Sequential"} or not isinstance(config, dict):
        raise KerasArchitectureImportError(
            "unsupported_model_type",
            "Only Functional and Sequential Keras architectures can be imported",
        )
    layers = config.get("layers")
    if not isinstance(layers, list) or not layers:
        raise KerasArchitectureImportError("missing_layers", "The Keras architecture does not contain a layer list")
    layer_by_name: dict[str, dict[str, Any]] = {}
    for layer in layers:
        if not isinstance(layer, dict) or not isinstance(layer.get("config"), dict):
            raise KerasArchitectureImportError("invalid_layer", "A serialized Keras layer is malformed")
        layer_name = layer.get("config", {}).get("name") or layer.get("name")
        if not isinstance(layer_name, str) or not layer_name:
            raise KerasArchitectureImportError("invalid_layer_name", "Every imported Keras layer must have a name")
        if layer_name in layer_by_name:
            raise KerasArchitectureImportError("duplicate_layer_name", f"Keras layer name '{layer_name}' is duplicated")
        if layer.get("class_name") not in SUPPORTED_LAYERS:
            raise KerasArchitectureImportError(
                "unsupported_layer",
                f"Keras layer '{layer.get('class_name')}' is not in the safe import allowlist",
            )
        layer_by_name[layer_name] = layer

    node_ids = _unique_ids(layer_by_name)
    nodes: list[NodeInstance] = []
    edges: list[dict[str, str]] = []
    for index, (layer_name, layer) in enumerate(layer_by_name.items()):
        definition_id, parameters = _map_layer(layer["class_name"], layer["config"])
        node_id = node_ids[layer_name]
        nodes.append(
            NodeInstance(
                id=node_id,
                definition_id=definition_id,
                label=layer_name,
                position=Position(x=float(index * 280), y=80.0),
                parameters=parameters,
            )
        )
        inbound_names = _inbound_names(layer.get("inbound_nodes", []))
        if class_name == "Sequential" and not inbound_names and index > 0:
            inbound_names = [list(layer_by_name)[index - 1]]
        for inbound_name in inbound_names:
            if inbound_name not in node_ids:
                raise KerasArchitectureImportError("unknown_connection", f"Keras layer '{layer_name}' references unknown layer '{inbound_name}'")
            target_port_id = "inputs" if definition_id in {"merge.add", "merge.concatenate"} else "input"
            if definition_id == "attention.multi_head":
                target_port_id = "query" if not any(edge["target_node_id"] == node_id for edge in edges) else "context"
            edges.append(
                {
                    "id": f"edge-{len(edges) + 1}",
                    "source_node_id": node_ids[inbound_name],
                    "source_port_id": "output",
                    "target_node_id": node_id,
                    "target_port_id": target_port_id,
                }
            )

    output_names = _model_output_names(config, class_name, list(layer_by_name))
    for output_index, output_name in enumerate(output_names):
        if output_name not in node_ids:
            raise KerasArchitectureImportError("unknown_output", f"Keras output references unknown layer '{output_name}'")
        source = layer_by_name[output_name]
        output_id = _unique_id(f"{node_ids[output_name]}_output", set(node_ids.values()) | {item.id for item in nodes})
        definition_id = "output.classification" if source["config"].get("activation") in {"softmax", "sigmoid"} else "output.regression"
        nodes.append(
            NodeInstance(
                id=output_id,
                definition_id=definition_id,
                label=f"{output_name} output",
                position=Position(x=float((len(layer_by_name) + output_index) * 280), y=80.0),
                parameters={},
            )
        )
        edges.append(
            {
                "id": f"edge-{len(edges) + 1}",
                "source_node_id": node_ids[output_name],
                "source_port_id": "output",
                "target_node_id": output_id,
                "target_port_id": "input",
            }
        )
    return ModelGraph(nodes=nodes, edges=[EdgeInstance(**edge) for edge in edges]), str(config.get("name") or "Imported Keras architecture")


###############################################################################
def _map_layer(class_name: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if class_name == "InputLayer":
        raw_shape = config.get("batch_shape") or config.get("batch_input_shape") or config.get("input_shape")
        shape = _shape(raw_shape)
        if shape and shape[0] is not None:
            shape = [None, *shape]
        if not shape:
            raise KerasArchitectureImportError("missing_input_shape", "An InputLayer must declare a deterministic shape")
        definition_id = "input.image" if len(shape) >= 4 else "input.sequence" if len(shape) == 3 else "input.tensor"
        return definition_id, {"shape": shape, "dtype": str(config.get("dtype") or "float32")}
    mappings: dict[str, tuple[str, tuple[str, ...]]] = {
        "Dense": ("layer.dense", ("units", "activation", "use_bias")),
        "Activation": ("layer.activation", ("activation",)),
        "Dropout": ("layer.dropout", ("rate",)),
        "Flatten": ("layer.flatten", ("data_format",)),
        "BatchNormalization": ("layer.batch_norm", ("momentum", "epsilon")),
        "LayerNormalization": ("layer.layer_norm", ("epsilon",)),
        "Embedding": ("layer.embedding", ("input_dim", "output_dim")),
        "Conv2D": ("conv.conv2d", ("filters", "kernel_size", "padding", "activation", "data_format", "use_bias")),
        "MaxPooling2D": ("conv.max_pool2d", ("pool_size",)),
        "LSTM": ("recurrent.lstm", ("units", "return_sequences")),
        "GRU": ("recurrent.gru", ("units", "return_sequences")),
        "MultiHeadAttention": ("attention.multi_head", ("num_heads", "key_dim", "dropout")),
        "Add": ("merge.add", ()),
        "Concatenate": ("merge.concatenate", ("axis",)),
    }
    definition_id, keys = mappings[class_name]
    parameters: dict[str, Any] = {}
    for key in keys:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, (str, int, float, bool)) or value is None or _is_scalar_list(value):
            target_key = {"input_dim": "vocabulary_size", "output_dim": "embedding_dim"}.get(key, key)
            parameters[target_key] = value
        else:
            raise KerasArchitectureImportError("unsupported_parameter", f"Keras layer parameter '{class_name}.{key}' is not a safe scalar value")
    return definition_id, parameters


###############################################################################
def _shape(value: Any) -> list[int | None] | None:
    if not _is_scalar_list(value) or not value:
        return None
    shape: list[int | None] = []
    for item in value:
        if item is None:
            shape.append(None)
        elif isinstance(item, int) and not isinstance(item, bool) and item > 0:
            shape.append(item)
        else:
            raise KerasArchitectureImportError("unsupported_shape", "Keras input shapes must contain positive integers or null")
    return shape


###############################################################################
def _is_scalar_list(value: Any) -> bool:
    return isinstance(value, list) and all(item is None or isinstance(item, (str, int, float, bool)) for item in value)


###############################################################################
def _unique_ids(layers: dict[str, dict[str, Any]]) -> dict[str, str]:
    used: set[str] = set()
    result: dict[str, str] = {}
    for name in layers:
        result[name] = _unique_id(name, used)
        used.add(result[name])
    return result


###############################################################################
def _unique_id(value: str, used: set[str]) -> str:
    candidate = _ID_PATTERN.sub("_", value).strip("_.-") or "node"
    if not candidate[0].isalpha():
        candidate = f"node_{candidate}"
    base = candidate[:116]
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base[: max(1, 120 - len(str(suffix)) - 1)]}_{suffix}"
        suffix += 1
    return candidate


###############################################################################
def _visit_inbound_names(item: Any, names: list[str]) -> None:
    if isinstance(item, dict):
        config = item.get("config")
        history = config.get("keras_history") if isinstance(config, dict) else None
        if isinstance(history, list) and history and isinstance(history[0], str):
            names.append(history[0])
            return
        for child in item.values():
            _visit_inbound_names(child, names)
    elif isinstance(item, list):
        if item and isinstance(item[0], str) and len(item) >= 3:
            names.append(item[0])
            return
        for child in item:
            _visit_inbound_names(child, names)


###############################################################################
def _inbound_names(value: Any) -> list[str]:
    names: list[str] = []
    _visit_inbound_names(value, names)
    return list(dict.fromkeys(names))


###############################################################################
def _model_output_names(config: dict[str, Any], class_name: str, layer_names: list[str]) -> list[str]:
    raw = config.get("output_layers")
    names = _inbound_names(raw) if raw is not None else []
    if names:
        return names
    if class_name == "Sequential" and layer_names:
        return [layer_names[-1]]
    return [layer_names[-1]]
