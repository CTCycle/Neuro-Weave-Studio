# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOperatorIssue=false

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import keras
from keras import layers

from domain.graph import ModelGraph
from validation.graph_utils import incoming_edges, topological_order


@keras.saving.register_keras_serializable(package="NeuroWeave")
class SinusoidalPositionalEncoding(layers.Layer):
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
                name=node.id,
            )(source)
        elif definition == "layer.activation":
            tensors[node_id] = layers.Activation(
                params.get("activation", "relu"), name=node.id
            )(source)
        elif definition == "layer.flatten":
            tensors[node_id] = layers.Flatten(name=node.id)(source)
        elif definition == "layer.dropout":
            tensors[node_id] = layers.Dropout(params.get("rate", 0.5), name=node.id)(source)
        elif definition == "layer.batch_norm":
            tensors[node_id] = layers.BatchNormalization(name=node.id)(source)
        elif definition == "layer.layer_norm":
            tensors[node_id] = layers.LayerNormalization(name=node.id)(source)
        elif definition == "layer.embedding":
            tensors[node_id] = layers.Embedding(
                params.get("vocabulary_size", 10000),
                params.get("embedding_dim", 128),
                name=node.id,
            )(source)
        elif definition == "conv.conv2d":
            tensors[node_id] = layers.Conv2D(
                params.get("filters", 32),
                params.get("kernel_size", [3, 3]),
                padding=params.get("padding", "same"),
                activation=params.get("activation", "linear"),
                name=node.id,
            )(source)
        elif definition == "conv.max_pool2d":
            tensors[node_id] = layers.MaxPooling2D(
                pool_size=params.get("pool_size", [2, 2]), name=node.id
            )(source)
        elif definition == "recurrent.lstm":
            tensors[node_id] = layers.LSTM(
                params.get("units", 64),
                return_sequences=params.get("return_sequences", False),
                name=node.id,
            )(source)
        elif definition == "recurrent.gru":
            tensors[node_id] = layers.GRU(
                params.get("units", 64),
                return_sequences=params.get("return_sequences", False),
                name=node.id,
            )(source)
        elif definition in {"attention.multi_head", "attention.self"}:
            query = inputs[0]
            value = inputs[1] if len(inputs) > 1 else query
            tensors[node_id] = layers.MultiHeadAttention(
                num_heads=params.get("num_heads", 8),
                key_dim=params.get("key_dim", 64),
                dropout=params.get("dropout", 0.0),
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
            raise RuntimeError(f"Binary export does not support node {definition}")

    inputs = [tensors[node.id] for node in graph.nodes if node.definition_id.startswith("input.")]
    outputs = [tensors[node.id] for node in graph.nodes if node.definition_id.startswith("output.")]
    return keras.Model(inputs=inputs, outputs=outputs, name="neuro_weave_model")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--destination", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    backend = keras.backend.backend()
    if arguments.probe:
        print(backend)
        return
    if arguments.destination is None:
        raise RuntimeError("A destination is required for binary export")
    graph = ModelGraph.model_validate_json(sys.stdin.read())
    model = _build_model(graph)
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(arguments.destination)


if __name__ == "__main__":
    main()
