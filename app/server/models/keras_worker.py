# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOperatorIssue=false

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import keras
from keras import layers

from domain.graph import ModelGraph
from validation.graph_utils import incoming_edges, topological_order

###############################################################################
def _option(value: Any) -> Any:
    return None if value in (None, "", "none") else value

###############################################################################
@keras.saving.register_keras_serializable(package="NeuroWeave")
class SinusoidalPositionalEncoding(layers.Layer):

    # -------------------------------------------------------------------------
    def call(self, inputs: Any) -> Any:
        length = keras.ops.shape(inputs)[1]
        depth = keras.ops.shape(inputs)[2]
        positions = keras.ops.cast(keras.ops.arange(length)[:, None], inputs.dtype)
        dimensions = keras.ops.cast(keras.ops.arange(depth)[None, :], inputs.dtype)
        depth_value = keras.ops.cast(depth, inputs.dtype)
        rates = keras.ops.power(
            10000.0,
            -(2.0 * keras.ops.floor(dimensions / 2.0)) / depth_value,
        )
        angles = positions * rates
        even = keras.ops.cast(
            keras.ops.mod(keras.ops.arange(depth), 2) == 0,
            inputs.dtype,
        )[None, :]
        encoding = keras.ops.sin(angles) * even + keras.ops.cos(angles) * (1.0 - even)
        return inputs + encoding[None, :, :]

###############################################################################
def _build_model(graph: ModelGraph) -> keras.Model:
    nodes = {node.id: node for node in graph.nodes}
    inbound = incoming_edges(graph)
    tensors: dict[str, Any] = {}
    for node_id in topological_order(graph):
        node = nodes[node_id]
        inputs = [tensors[edge.source_node_id] for edge in inbound.get(node_id, [])]
        source = inputs[0] if inputs else None
        params = node.parameters
        definition = node.definition_id
        if definition.startswith("input."):
            shape = list(params.get("shape", [None, 1]))
            feature_shape = shape[1:] if shape and shape[0] is None else shape
            tensors[node_id] = keras.Input(shape=feature_shape, name=node.id)
        elif definition == "layer.dense":
            tensors[node_id] = layers.Dense(
                params.get("units", 32),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=_option(params.get("kernel_initializer", "glorot_uniform")),
                bias_initializer=_option(params.get("bias_initializer", "zeros")),
                kernel_regularizer=_option(params.get("kernel_regularizer", "none")),
                bias_regularizer=_option(params.get("bias_regularizer", "none")),
                activity_regularizer=_option(params.get("activity_regularizer", "none")),
                kernel_constraint=_option(params.get("kernel_constraint", "none")),
                bias_constraint=_option(params.get("bias_constraint", "none")),
                name=node.id,
            )(source)
        elif definition == "layer.activation":
            tensors[node_id] = layers.Activation(
                params.get("activation", "relu"), name=node.id
            )(source)
        elif definition == "layer.flatten":
            tensors[node_id] = layers.Flatten(name=node.id)(source)
        elif definition == "layer.dropout":
            tensors[node_id] = layers.Dropout(
                params.get("rate", 0.5), seed=params.get("seed"), name=node.id
            )(source)
        elif definition == "layer.batch_norm":
            tensors[node_id] = layers.BatchNormalization(
                axis=params.get("axis", -1),
                momentum=params.get("momentum", 0.99),
                epsilon=params.get("epsilon", 0.001),
                center=params.get("center", True),
                scale=params.get("scale", True),
                beta_initializer=_option(params.get("beta_initializer", "zeros")),
                gamma_initializer=_option(params.get("gamma_initializer", "ones")),
                moving_mean_initializer=_option(params.get("moving_mean_initializer", "zeros")),
                moving_variance_initializer=_option(params.get("moving_variance_initializer", "ones")),
                beta_regularizer=_option(params.get("beta_regularizer", "none")),
                gamma_regularizer=_option(params.get("gamma_regularizer", "none")),
                beta_constraint=_option(params.get("beta_constraint", "none")),
                gamma_constraint=_option(params.get("gamma_constraint", "none")),
                synchronized=params.get("synchronized", False),
                name=node.id,
            )(source)
        elif definition == "layer.layer_norm":
            tensors[node_id] = layers.LayerNormalization(
                axis=params.get("axis", [-1]),
                epsilon=params.get("epsilon", 0.001),
                center=params.get("center", True),
                scale=params.get("scale", True),
                beta_initializer=_option(params.get("beta_initializer", "zeros")),
                gamma_initializer=_option(params.get("gamma_initializer", "ones")),
                beta_regularizer=_option(params.get("beta_regularizer", "none")),
                gamma_regularizer=_option(params.get("gamma_regularizer", "none")),
                beta_constraint=_option(params.get("beta_constraint", "none")),
                gamma_constraint=_option(params.get("gamma_constraint", "none")),
                name=node.id,
            )(source)
        elif definition == "layer.embedding":
            tensors[node_id] = layers.Embedding(
                params.get("vocabulary_size", 10000),
                params.get("embedding_dim", 128),
                embeddings_initializer=_option(params.get("embeddings_initializer", "uniform")),
                embeddings_regularizer=_option(params.get("embeddings_regularizer", "none")),
                activity_regularizer=_option(params.get("activity_regularizer", "none")),
                embeddings_constraint=_option(params.get("embeddings_constraint", "none")),
                mask_zero=params.get("mask_zero", False),
                lora_rank=params.get("lora_rank"),
                name=node.id,
            )(source)
        elif definition == "conv.conv2d":
            tensors[node_id] = layers.Conv2D(
                params.get("filters", 32),
                params.get("kernel_size", [3, 3]),
                strides=params.get("strides", [1, 1]),
                padding=params.get("padding", "same"),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1, 1]),
                groups=params.get("groups", 1),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=_option(params.get("kernel_initializer", "glorot_uniform")),
                bias_initializer=_option(params.get("bias_initializer", "zeros")),
                kernel_regularizer=_option(params.get("kernel_regularizer", "none")),
                bias_regularizer=_option(params.get("bias_regularizer", "none")),
                activity_regularizer=_option(params.get("activity_regularizer", "none")),
                kernel_constraint=_option(params.get("kernel_constraint", "none")),
                bias_constraint=_option(params.get("bias_constraint", "none")),
                name=node.id,
            )(source)
        elif definition == "conv.max_pool2d":
            tensors[node_id] = layers.MaxPooling2D(
                pool_size=params.get("pool_size", [2, 2]),
                strides=params.get("strides"),
                padding=params.get("padding", "valid"),
                data_format=params.get("data_format", "channels_last"),
                name=node.id,
            )(source)
        elif definition == "recurrent.lstm":
            tensors[node_id] = layers.LSTM(
                params.get("units", 64),
                activation=params.get("activation", "tanh"),
                recurrent_activation=params.get("recurrent_activation", "sigmoid"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=_option(params.get("kernel_initializer", "glorot_uniform")),
                recurrent_initializer=_option(params.get("recurrent_initializer", "orthogonal")),
                bias_initializer=_option(params.get("bias_initializer", "zeros")),
                unit_forget_bias=params.get("unit_forget_bias", True),
                dropout=params.get("dropout", 0.0),
                recurrent_dropout=params.get("recurrent_dropout", 0.0),
                return_sequences=params.get("return_sequences", False),
                return_state=params.get("return_state", False),
                go_backwards=params.get("go_backwards", False),
                stateful=params.get("stateful", False),
                unroll=params.get("unroll", False),
                seed=params.get("seed"),
                name=node.id,
            )(source)
        elif definition == "recurrent.gru":
            tensors[node_id] = layers.GRU(
                params.get("units", 64),
                activation=params.get("activation", "tanh"),
                recurrent_activation=params.get("recurrent_activation", "sigmoid"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=_option(params.get("kernel_initializer", "glorot_uniform")),
                recurrent_initializer=_option(params.get("recurrent_initializer", "orthogonal")),
                bias_initializer=_option(params.get("bias_initializer", "zeros")),
                reset_after=params.get("reset_after", True),
                dropout=params.get("dropout", 0.0),
                recurrent_dropout=params.get("recurrent_dropout", 0.0),
                return_sequences=params.get("return_sequences", False),
                return_state=params.get("return_state", False),
                go_backwards=params.get("go_backwards", False),
                stateful=params.get("stateful", False),
                unroll=params.get("unroll", False),
                seed=params.get("seed"),
                name=node.id,
            )(source)
        elif definition in {"attention.multi_head", "attention.self"}:
            query = inputs[0]
            value = inputs[1] if len(inputs) > 1 else query
            tensors[node_id] = layers.MultiHeadAttention(
                num_heads=params.get("num_heads", 8),
                key_dim=params.get("key_dim", 64),
                value_dim=params.get("value_dim"),
                dropout=params.get("dropout", 0.0),
                use_bias=params.get("use_bias", True),
                output_shape=params.get("output_shape"),
                attention_axes=params.get("attention_axes"),
                kernel_initializer=_option(params.get("kernel_initializer", "glorot_uniform")),
                bias_initializer=_option(params.get("bias_initializer", "zeros")),
                name=node.id,
            )(query, value)
        elif definition in {"merge.add", "merge.residual", "merge.skip"}:
            tensors[node_id] = layers.Add(name=node.id)(inputs)
        elif definition == "merge.concatenate":
            tensors[node_id] = layers.Concatenate(
                axis=params.get("axis", -1), name=node.id
            )(inputs)
        elif definition == "transformer.positional_encoding":
            tensors[node_id] = SinusoidalPositionalEncoding(name=node.id)(source)
        elif definition == "transformer.encoder_block":
            heads = int(params.get("num_heads", 8))
            key_dim = int(params.get("key_dim", 64))
            dropout = float(params.get("dropout", 0.0))
            model_dim = heads * key_dim
            attention = layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=key_dim,
                dropout=dropout,
                name=f"{node.id}_attention",
            )(source, source)
            residual = layers.LayerNormalization(name=f"{node.id}_attention_norm")(
                layers.Add()([source, attention])
            )
            feed_forward = layers.Dense(
                model_dim * 4,
                activation="gelu",
                name=f"{node.id}_ffn_expand",
            )(residual)
            feed_forward = layers.Dense(
                model_dim,
                name=f"{node.id}_ffn_project",
            )(feed_forward)
            feed_forward = layers.Dropout(
                dropout,
                name=f"{node.id}_ffn_dropout",
            )(feed_forward)
            tensors[node_id] = layers.LayerNormalization(
                name=f"{node.id}_output_norm"
            )(layers.Add()([residual, feed_forward]))
        elif definition.startswith("output."):
            tensors[node_id] = source
        else:
            raise RuntimeError(f"Keras worker does not support node {definition}")

    inputs = [tensors[node.id] for node in graph.nodes if node.definition_id.startswith("input.")]
    outputs = [tensors[node.id] for node in graph.nodes if node.definition_id.startswith("output.")]
    return keras.Model(inputs=inputs, outputs=outputs, name="neuro_weave_model")

###############################################################################
def _backend_capabilities() -> dict[str, object]:
    backend = str(keras.backend.backend())
    accelerator = "cpu"
    device_name = "CPU"
    if backend == "torch":
        torch = import_module("torch")

        if bool(torch.cuda.is_available()):
            accelerator = "cuda"
            device_name = str(torch.cuda.get_device_name(0))
    elif backend == "tensorflow":
        tf = import_module("tensorflow")

        devices = tf.config.list_physical_devices("GPU")
        if devices:
            accelerator = "cuda"
            device_name = str(devices[0].name)
    elif backend == "jax":
        jax = import_module("jax")

        devices = list(jax.devices())
        if devices:
            platform = str(devices[0].platform)
            if platform not in {"cpu", ""}:
                accelerator = platform
            device_name = str(devices[0])
    return {
        "backend": backend,
        "available": True,
        "accelerator": accelerator,
        "device_name": device_name,
    }

###############################################################################
def _concrete_shape(tensor: Any) -> list[int]:
    shape = getattr(tensor, "shape", ())
    result: list[int] = []
    for dimension in shape:
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            value = 1
        result.append(value if value > 0 else 1)
    return result

###############################################################################
def _dtype_name(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", "float32")
    name = str(getattr(dtype, "name", dtype))
    return name.rsplit(".", 1)[-1]

###############################################################################
def _smoke_test(graph: ModelGraph) -> dict[str, object]:
    started = time.perf_counter()
    capabilities = _backend_capabilities()
    use_gpu = capabilities.get("accelerator") in {"cuda", "gpu", "rocm"}
    device_scope = keras.device("gpu:0") if use_gpu else contextlib.nullcontext()
    with device_scope:
        model = _build_model(graph)
        input_values = [
            keras.ops.zeros(_concrete_shape(tensor), dtype=_dtype_name(tensor))
            for tensor in model.inputs
        ]
        model_input: Any = input_values[0] if len(input_values) == 1 else input_values
        output_values = model(model_input, training=False)

    actual_outputs = list(output_values) if isinstance(output_values, (list, tuple)) else [output_values]
    input_nodes = [node for node in graph.nodes if node.definition_id.startswith("input.")]
    output_nodes = [node for node in graph.nodes if node.definition_id.startswith("output.")]
    inputs = [
        {
            "node_id": node.id,
            "shape": _concrete_shape(tensor),
            "dtype": _dtype_name(tensor),
        }
        for node, tensor in zip(input_nodes, input_values, strict=False)
    ]
    outputs = [
        {
            "node_id": node.id,
            "shape": _concrete_shape(tensor),
            "dtype": _dtype_name(tensor),
        }
        for node, tensor in zip(output_nodes, actual_outputs, strict=False)
    ]
    return {
        "status": "succeeded",
        "accelerator": capabilities["accelerator"],
        "device_name": capabilities["device_name"],
        "inputs": inputs,
        "outputs": outputs,
        "parameter_count": int(model.count_params()),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }

###############################################################################
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--destination", type=Path)
    return parser.parse_args()

###############################################################################
def main() -> None:
    arguments = _parse_arguments()
    if arguments.probe:
        print(json.dumps(_backend_capabilities()))
        return
    if arguments.smoke_test:
        graph = ModelGraph.model_validate_json(sys.stdin.read())
        print(json.dumps(_smoke_test(graph)))
        return
    if arguments.destination is None:
        raise RuntimeError("A destination is required for binary export")
    graph = ModelGraph.model_validate_json(sys.stdin.read())
    model = _build_model(graph)
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(arguments.destination)


if __name__ == "__main__":
    main()
