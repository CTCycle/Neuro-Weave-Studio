# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOperatorIssue=false

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import keras
from keras import layers

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import apply_runtime_operation
from domain.graph import ModelGraph
from models.keras_layers import SinusoidalPositionalEncoding
from models.keras_runtime import layer_option, prepare_runtime
from validation.graph_utils import ordered_incoming_edges, topological_order

###############################################################################
def _checkpoint(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()

###############################################################################
def _keras_operation_functions(catalog: NodeCatalog) -> dict[str, Callable[..., Any]]:
    symbols = {"keras.ops.shape"}
    for manifest in catalog.manifests:
        mapping = manifest.implementations.get("keras")
        if manifest.operation is None or mapping is None:
            continue
        if mapping.symbol is not None:
            symbols.add(mapping.symbol)
        symbols.update(mapping.variant_symbols.values())
    functions: dict[str, Callable[..., Any]] = {}
    for symbol in symbols:
        prefix, _, name = symbol.rpartition(".")
        if prefix != "keras.ops" or not name.isidentifier():
            continue
        function = getattr(keras.ops, name, None)
        if callable(function):
            functions[symbol] = function
    return functions

###############################################################################
def _transformer_encoder_block(source: Any, node_id: str, parameters: dict[str, Any]) -> Any:
    source_dimension = source.shape[-1]
    if source_dimension is None:
        raise RuntimeError("Transformer encoder requires a known embedding dimension")
    layer_count = int(parameters.get("layers", 1))
    heads = int(parameters.get("num_heads", 8))
    key_dim = int(parameters.get("key_dim", 64))
    dropout = float(parameters.get("dropout", 0.0))
    hidden_dimension = int(source_dimension) * int(parameters.get("ffn_multiplier", 4))
    activation = parameters.get("activation", "gelu")
    pre_norm = bool(parameters.get("pre_norm", True))
    current = source
    for index in range(layer_count):
        suffix = f"_{index + 1}" if layer_count > 1 else ""
        attention_name = f"{node_id}{suffix}_attention"
        attention_norm_name = f"{node_id}{suffix}_attention_norm"
        ffn_norm_name = f"{node_id}{suffix}_ffn_norm"
        if pre_norm:
            attention_input = layers.LayerNormalization(name=attention_norm_name)(current)
            attention = layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=key_dim,
                dropout=dropout,
                name=attention_name,
            )(attention_input, attention_input)
            current = layers.Add(name=f"{node_id}{suffix}_attention_residual")(
                [current, attention]
            )
            feed_forward_input = layers.LayerNormalization(name=ffn_norm_name)(current)
        else:
            attention = layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=key_dim,
                dropout=dropout,
                name=attention_name,
            )(current, current)
            current = layers.LayerNormalization(name=attention_norm_name)(
                layers.Add(name=f"{node_id}{suffix}_attention_residual")([current, attention])
            )
            feed_forward_input = current
        feed_forward = layers.Dense(
            hidden_dimension,
            activation=activation,
            name=f"{node_id}{suffix}_ffn_expand",
        )(feed_forward_input)
        feed_forward = layers.Dense(
            int(source_dimension),
            name=f"{node_id}{suffix}_ffn_project",
        )(feed_forward)
        feed_forward = layers.Dropout(
            dropout,
            name=f"{node_id}{suffix}_ffn_dropout",
        )(feed_forward)
        current = layers.Add(name=f"{node_id}{suffix}_ffn_residual")(
            [current, feed_forward]
        )
        if not pre_norm:
            current = layers.LayerNormalization(name=f"{node_id}{suffix}_output_norm")(current)
    return current

###############################################################################
def build_model(
    graph: ModelGraph,
    catalog: NodeCatalog | None = None,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> keras.Model:
    _checkpoint(checkpoint)
    prepare_runtime(graph)
    catalog = catalog or get_node_catalog()
    nodes = {node.id: node for node in graph.nodes}
    tensors: dict[str, Any] = {}
    for node_id in topological_order(graph):
        _checkpoint(checkpoint)
        node = nodes[node_id]
        manifest = catalog.get(node.definition_id, node.definition_version)
        inputs = [
            tensors[edge.source_node_id]
            for edge in ordered_incoming_edges(graph, node_id, manifest)
        ]
        source = inputs[0] if inputs else None
        params = node.parameters
        definition = node.definition_id
        if manifest.operation is not None:
            tensors[node_id] = apply_runtime_operation(
                manifest,
                "keras",
                inputs,
                params,
                _keras_operation_functions(catalog),
            )
        elif definition.startswith("input."):
            shape = list(params.get("shape", [None, 1]))
            feature_shape = shape[1:] if shape and shape[0] is None else shape
            tensors[node_id] = keras.Input(
                shape=feature_shape,
                dtype=params.get("dtype", "float32"),
                name=node.id,
            )
        elif definition == "layer.dense":
            tensors[node_id] = layers.Dense(
                params.get("units", 32),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=layer_option(params, "kernel_initializer", graph, "glorot_uniform"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                name=node.id,
            )(source)
        elif definition == "layer.activation":
            tensors[node_id] = layers.Activation(
                params.get("activation", "relu"), name=node.id
            )(source)
        elif definition == "layer.flatten":
            tensors[node_id] = layers.Flatten(
                data_format=params.get("data_format", "channels_last"),
                name=node.id,
            )(source)
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
                beta_initializer=layer_option(params, "beta_initializer", graph, "zeros"),
                gamma_initializer=layer_option(params, "gamma_initializer", graph, "ones"),
                moving_mean_initializer=layer_option(params, "moving_mean_initializer", graph, "zeros"),
                moving_variance_initializer=layer_option(params, "moving_variance_initializer", graph, "ones"),
                beta_regularizer=layer_option(params, "beta_regularizer", graph, "none"),
                gamma_regularizer=layer_option(params, "gamma_regularizer", graph, "none"),
                beta_constraint=layer_option(params, "beta_constraint", graph, "none"),
                gamma_constraint=layer_option(params, "gamma_constraint", graph, "none"),
                synchronized=params.get("synchronized", False),
                name=node.id,
            )(source)
        elif definition == "layer.layer_norm":
            tensors[node_id] = layers.LayerNormalization(
                axis=params.get("axis", [-1]),
                epsilon=params.get("epsilon", 0.001),
                center=params.get("center", True),
                scale=params.get("scale", True),
                beta_initializer=layer_option(params, "beta_initializer", graph, "zeros"),
                gamma_initializer=layer_option(params, "gamma_initializer", graph, "ones"),
                beta_regularizer=layer_option(params, "beta_regularizer", graph, "none"),
                gamma_regularizer=layer_option(params, "gamma_regularizer", graph, "none"),
                beta_constraint=layer_option(params, "beta_constraint", graph, "none"),
                gamma_constraint=layer_option(params, "gamma_constraint", graph, "none"),
                name=node.id,
            )(source)
        elif definition == "layer.embedding":
            tensors[node_id] = layers.Embedding(
                params.get("vocabulary_size", 10000),
                params.get("embedding_dim", 128),
                embeddings_initializer=layer_option(params, "embeddings_initializer", graph, "uniform"),
                embeddings_regularizer=layer_option(params, "embeddings_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                embeddings_constraint=layer_option(params, "embeddings_constraint", graph, "none"),
                mask_zero=params.get("mask_zero", False),
                lora_rank=params.get("lora_rank"),
                name=node.id,
            )(source)
        elif definition == "conv.conv1d":
            tensors[node_id] = layers.Conv1D(
                params.get("filters", 32),
                params.get("kernel_size", [3]),
                strides=params.get("strides", [1]),
                padding=params.get("padding", "same"),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1]),
                groups=params.get("groups", 1),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=layer_option(params, "kernel_initializer", graph, "glorot_uniform"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
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
                kernel_initializer=layer_option(params, "kernel_initializer", graph, "glorot_uniform"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
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
                kernel_initializer=layer_option(params, "kernel_initializer", graph, "glorot_uniform"),
                recurrent_initializer=layer_option(params, "recurrent_initializer", graph, "orthogonal"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                unit_forget_bias=params.get("unit_forget_bias", True),
                dropout=params.get("dropout", 0.0),
                recurrent_dropout=params.get("recurrent_dropout", 0.0),
                return_sequences=params.get("return_sequences", False),
                return_state=False,
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
                kernel_initializer=layer_option(params, "kernel_initializer", graph, "glorot_uniform"),
                recurrent_initializer=layer_option(params, "recurrent_initializer", graph, "orthogonal"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                reset_after=params.get("reset_after", True),
                dropout=params.get("dropout", 0.0),
                recurrent_dropout=params.get("recurrent_dropout", 0.0),
                return_sequences=params.get("return_sequences", False),
                return_state=False,
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
                kernel_initializer=layer_option(params, "kernel_initializer", graph, "glorot_uniform"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
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
            tensors[node_id] = _transformer_encoder_block(source, node.id, params)
        elif definition.startswith("output."):
            tensors[node_id] = layers.Activation("linear", name=node.id)(source)
        else:
            raise RuntimeError(f"Keras worker does not support node {definition}")

    inputs = [tensors[node.id] for node in graph.nodes if node.definition_id.startswith("input.")]
    outputs = [tensors[node.id] for node in graph.nodes if node.definition_id.startswith("output.")]
    return keras.Model(inputs=inputs, outputs=outputs, name="neuro_weave_model")
