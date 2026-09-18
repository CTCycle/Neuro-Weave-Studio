# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOperatorIssue=false

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import keras
from keras import layers

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import apply_runtime_operation
from domain.graph import ModelGraph
from models.keras_layers import GeluActivation, SinusoidalPositionalEncoding
from models.keras_runtime import layer_option, prepare_runtime
from validation.graph_utils import ordered_incoming_edges, topological_order
from validation.output_resolution import input_nodes, resolved_outputs
from validation.shape_inference import infer_shapes


###############################################################################
class GeneratedConstant(layers.Layer):
    def __init__(self, shape: list[int], value: Any, dtype: str, **kwargs: Any) -> None:
        super().__init__(dtype=dtype, trainable=False, **kwargs)
        self._source_shape = tuple(shape)
        self._value = value

    def call(self, anchor: Any) -> Any:
        batch = keras.ops.shape(anchor)[0]
        value = keras.ops.full(self._source_shape, self._value, dtype=self.compute_dtype)
        value = keras.ops.expand_dims(value, axis=0)
        return keras.ops.repeat(value, batch, axis=0)


###############################################################################
class GeneratedParameter(layers.Layer):
    def __init__(
        self,
        shape: list[int],
        initializer: Any,
        dtype: str,
        trainable: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype=dtype, trainable=trainable, **kwargs)
        self._source_shape = tuple(shape)
        self._initializer = keras.initializers.get(initializer)
        self._parameter_trainable = trainable

    def build(self, input_shape: Any) -> None:
        del input_shape
        self.value = self.add_weight(
            name="value",
            shape=self._source_shape,
            initializer=self._initializer,
            trainable=self._parameter_trainable,
            dtype=self.compute_dtype,
        )

    def call(self, anchor: Any) -> Any:
        batch = keras.ops.shape(anchor)[0]
        value = keras.ops.expand_dims(self.value, axis=0)
        return keras.ops.repeat(value, batch, axis=0)


###############################################################################
def _checkpoint(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()

###############################################################################
def _base_layer_kwargs(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: parameters[key]
        for key in ("dtype", "trainable", "autocast")
        if key in parameters and parameters[key] is not None
    }

###############################################################################
def _condition_matches(condition: dict[str, Any], parameters: dict[str, Any]) -> bool:
    for parameter_id, expected in condition.items():
        actual = parameters.get(parameter_id)
        if isinstance(expected, dict):
            if "equals" in expected and actual != expected["equals"]:
                return False
            if "not_equals" in expected and actual == expected["not_equals"]:
                return False
            if "in" in expected and actual not in expected["in"]:
                return False
            if "not_in" in expected and actual in expected["not_in"]:
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


###############################################################################
def _enabled_output_ports(manifest: Any, parameters: dict[str, Any]) -> list[Any]:
    return [
        port
        for port in manifest.outputs
        if _condition_matches(port.enabled_when, parameters)
    ]

###############################################################################
def _recurrent_layer(
    cell: str,
    parameters: dict[str, Any],
    graph: ModelGraph,
    *,
    name: str,
    return_sequences: bool | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        **_base_layer_kwargs(parameters),
        "activation": parameters.get("activation", "tanh"),
        "use_bias": parameters.get("use_bias", True),
        "kernel_initializer": layer_option(
            parameters, "kernel_initializer", graph, "glorot_uniform"
        ),
        "recurrent_initializer": layer_option(
            parameters, "recurrent_initializer", graph, "orthogonal"
        ),
        "bias_initializer": layer_option(
            parameters, "bias_initializer", graph, "zeros"
        ),
        "kernel_regularizer": layer_option(
            parameters, "kernel_regularizer", graph, "none"
        ),
        "recurrent_regularizer": layer_option(
            parameters, "recurrent_regularizer", graph, "none"
        ),
        "bias_regularizer": layer_option(
            parameters, "bias_regularizer", graph, "none"
        ),
        "activity_regularizer": layer_option(
            parameters, "activity_regularizer", graph, "none"
        ),
        "kernel_constraint": layer_option(
            parameters, "kernel_constraint", graph, "none"
        ),
        "recurrent_constraint": layer_option(
            parameters, "recurrent_constraint", graph, "none"
        ),
        "bias_constraint": layer_option(
            parameters, "bias_constraint", graph, "none"
        ),
        "dropout": parameters.get("dropout", 0.0),
        "recurrent_dropout": parameters.get("recurrent_dropout", 0.0),
        "return_sequences": parameters.get("return_sequences", False)
        if return_sequences is None
        else return_sequences,
        "return_state": parameters.get("return_state", False),
        "go_backwards": parameters.get("go_backwards", False),
        "stateful": parameters.get("stateful", False),
        "unroll": parameters.get("unroll", False),
        "seed": parameters.get("seed"),
    }
    if cell == "simple_rnn":
        return layers.SimpleRNN(parameters.get("units", 64), name=name, **kwargs)
    kwargs["recurrent_activation"] = parameters.get("recurrent_activation", "sigmoid")
    if cell == "gru":
        kwargs["reset_after"] = parameters.get("reset_after", True)
        kwargs["use_cudnn"] = parameters.get("use_cudnn", "auto")
        return layers.GRU(parameters.get("units", 64), name=name, **kwargs)
    kwargs["unit_forget_bias"] = parameters.get("unit_forget_bias", True)
    kwargs["use_cudnn"] = parameters.get("use_cudnn", "auto")
    return layers.LSTM(parameters.get("units", 64), name=name, **kwargs)

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
    base_kwargs = _base_layer_kwargs(parameters)
    current = source
    for index in range(layer_count):
        suffix = f"_{index + 1}" if layer_count > 1 else ""
        attention_name = f"{node_id}{suffix}_attention"
        attention_norm_name = f"{node_id}{suffix}_attention_norm"
        ffn_norm_name = f"{node_id}{suffix}_ffn_norm"
        if pre_norm:
            attention_input = layers.LayerNormalization(**base_kwargs, name=attention_norm_name)(current)
            attention = layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=key_dim,
                dropout=dropout,
                **base_kwargs,
                name=attention_name,
            )(attention_input, attention_input)
            current = layers.Add(name=f"{node_id}{suffix}_attention_residual")([current, attention])
            feed_forward_input = layers.LayerNormalization(**base_kwargs, name=ffn_norm_name)(current)
        else:
            attention = layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=key_dim,
                dropout=dropout,
                **base_kwargs,
                name=attention_name,
            )(current, current)
            current = layers.LayerNormalization(**base_kwargs, name=attention_norm_name)(
                layers.Add(name=f"{node_id}{suffix}_attention_residual")([current, attention])
            )
            feed_forward_input = current
        feed_forward = layers.Dense(
            hidden_dimension,
            activation=None if activation == "gelu" else activation,
            **base_kwargs,
            name=f"{node_id}{suffix}_ffn_expand",
        )(feed_forward_input)
        if activation == "gelu":
            feed_forward = GeluActivation(
                **base_kwargs,
                name=f"{node_id}{suffix}_ffn_activation",
            )(feed_forward)
        feed_forward = layers.Dense(
            int(source_dimension),
            **base_kwargs,
            name=f"{node_id}{suffix}_ffn_project",
        )(feed_forward)
        feed_forward = layers.Dropout(
            dropout,
            **base_kwargs,
            name=f"{node_id}{suffix}_ffn_dropout",
        )(feed_forward)
        current = layers.Add(name=f"{node_id}{suffix}_ffn_residual")([current, feed_forward])
        if not pre_norm:
            current = layers.LayerNormalization(
                **base_kwargs, name=f"{node_id}{suffix}_output_norm"
            )(current)
    return current

###############################################################################
def build_model(
    graph: ModelGraph,
    catalog: NodeCatalog | None = None,
    *,
    checkpoint: Callable[[], None] | None = None,
    trace_outputs: dict[str, dict[str, Any]] | None = None,
) -> keras.Model:
    _checkpoint(checkpoint)
    prepare_runtime(graph)
    catalog = catalog or get_node_catalog()
    shape_summaries = {
        summary.node_id: summary for summary in infer_shapes(graph, catalog)
    }
    nodes = {node.id: node for node in graph.nodes}
    tensors: dict[str, Any] = {}
    port_tensors: dict[tuple[str, str], Any] = {}
    runtime_inputs = input_nodes(graph, catalog)
    for runtime_input in runtime_inputs:
        shape = list(runtime_input.parameters.get("shape", [None, 1]))
        feature_shape = shape[1:] if shape and shape[0] is None else shape
        tensors[runtime_input.id] = keras.Input(
            shape=feature_shape,
            dtype=runtime_input.parameters.get("dtype", "float32"),
            name=runtime_input.id,
        )
    for node_id in topological_order(graph):
        _checkpoint(checkpoint)
        node = nodes[node_id]
        manifest = catalog.get(node.definition_id, node.definition_version)
        incoming_edges = ordered_incoming_edges(graph, node_id, manifest)
        inputs = [
            port_tensors[(edge.source_node_id, edge.source_port_id)]
            for edge in incoming_edges
        ]
        source = inputs[0] if inputs else None
        params = node.parameters
        definition = node.definition_id
        if manifest.semantic_role != "input" and source is None:
            raise RuntimeError(f"Node {node_id} requires an inbound connection")
        source = cast(Any, source)
        if definition in {"input.constant", "input.parameter"}:
            if not runtime_inputs:
                raise RuntimeError(
                    f"Generated source node {node_id} requires at least one runtime model input for batch materialization"
                )
            anchor = tensors[runtime_inputs[0].id]
            source_shape = params.get("shape", [1])
            if not isinstance(source_shape, list):
                raise RuntimeError(f"Generated source node {node_id} requires a list shape")
            integer_shape = [int(value) for value in source_shape]
            dtype = str(params.get("dtype", "float32"))
            if definition == "input.constant":
                tensors[node_id] = GeneratedConstant(
                    integer_shape,
                    params.get("value", 0),
                    dtype,
                    name=node.id,
                )(anchor)
            else:
                tensors[node_id] = GeneratedParameter(
                    integer_shape,
                    params.get("initializer", "zeros"),
                    dtype,
                    bool(params.get("trainable", True)),
                    name=node.id,
                )(anchor)
        elif manifest.operation is not None:
            if manifest.operation.shape_behavior == "reshape":
                tensors[node_id] = layers.Reshape(
                    target_shape=params.get("target_shape", [-1]), name=node.id
                )(source)
            elif manifest.operation.shape_behavior == "flatten":
                tensors[node_id] = layers.Flatten(name=node.id)(source)
            else:
                tensors[node_id] = apply_runtime_operation(
                    manifest,
                    "keras",
                    inputs,
                    params,
                    _keras_operation_functions(catalog),
                )
        elif manifest.semantic_role == "input":
            if node_id not in tensors:
                raise RuntimeError(f"Runtime input {node_id} was not materialized")
        elif definition == "layer.dense":
            tensors[node_id] = layers.Dense(
                params.get("units", 32),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=layer_option(
                    params, "kernel_initializer", graph, "glorot_uniform"
                ),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                lora_rank=params.get("lora_rank"),
                lora_alpha=params.get("lora_alpha"),
                quantization_config=params.get("quantization_config"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.activation":
            tensors[node_id] = layers.Activation(
                params.get("activation", "relu"), **_base_layer_kwargs(params), name=node.id
            )(source)
        elif definition == "layer.prelu":
            tensors[node_id] = layers.PReLU(
                alpha_initializer=layer_option(params, "alpha_initializer", graph, "zeros"),
                alpha_regularizer=layer_option(params, "alpha_regularizer", graph, "none"),
                alpha_constraint=layer_option(params, "alpha_constraint", graph, "none"),
                shared_axes=params.get("shared_axes"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.flatten":
            tensors[node_id] = layers.Flatten(
                data_format=params.get("data_format", "channels_last"),
                name=node.id,
            )(source)
        elif definition == "layer.dropout":
            tensors[node_id] = layers.Dropout(
                params.get("rate", 0.5),
                noise_shape=params.get("noise_shape"),
                seed=params.get("seed"),
                **_base_layer_kwargs(params),
                name=node.id,
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
                moving_mean_initializer=layer_option(
                    params, "moving_mean_initializer", graph, "zeros"
                ),
                moving_variance_initializer=layer_option(
                    params, "moving_variance_initializer", graph, "ones"
                ),
                beta_regularizer=layer_option(params, "beta_regularizer", graph, "none"),
                gamma_regularizer=layer_option(params, "gamma_regularizer", graph, "none"),
                beta_constraint=layer_option(params, "beta_constraint", graph, "none"),
                gamma_constraint=layer_option(params, "gamma_constraint", graph, "none"),
                renorm=params.get("renorm", False),
                renorm_clipping=params.get("renorm_clipping"),
                renorm_momentum=params.get("renorm_momentum", 0.99),
                synchronized=params.get("synchronized", False),
                **_base_layer_kwargs(params),
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
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.rms_norm":
            tensors[node_id] = layers.RMSNormalization(
                axis=params.get("axis", -1),
                epsilon=params.get("epsilon", 1e-6),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.unit_norm":
            tensors[node_id] = layers.UnitNormalization(
                axis=params.get("axis", -1),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.group_norm":
            tensors[node_id] = layers.GroupNormalization(
                groups=params.get("groups", 32),
                axis=params.get("axis", -1),
                epsilon=params.get("epsilon", 0.001),
                center=params.get("center", True),
                scale=params.get("scale", True),
                beta_initializer=layer_option(params, "beta_initializer", graph, "zeros"),
                gamma_initializer=layer_option(params, "gamma_initializer", graph, "ones"),
                beta_regularizer=layer_option(params, "beta_regularizer", graph, "none"),
                gamma_regularizer=layer_option(params, "gamma_regularizer", graph, "none"),
                beta_constraint=layer_option(params, "beta_constraint", graph, "none"),
                gamma_constraint=layer_option(params, "gamma_constraint", graph, "none"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.reshape":
            tensors[node_id] = layers.Reshape(
                tuple(params.get("target_shape", [-1])),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.permute":
            tensors[node_id] = layers.Permute(
                tuple(params.get("axes", [2, 1])),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "layer.repeat":
            repeats = int(params.get("repeats", 2))
            axis = int(params.get("axis", 1))
            if axis == 1:
                tensors[node_id] = layers.RepeatVector(
                    repeats,
                    **_base_layer_kwargs(params),
                    name=node.id,
                )(source)
            else:
                tensors[node_id] = keras.layers.Lambda(
                    lambda value, repeats=repeats, axis=axis: keras.ops.repeat(
                        value, repeats, axis=axis
                    ),
                    **_base_layer_kwargs(params),
                    name=node.id,
                )(source)
        elif definition == "layer.embedding":
            tensors[node_id] = layers.Embedding(
                params.get("vocabulary_size", 10000),
                params.get("embedding_dim", 128),
                embeddings_initializer=layer_option(
                    params, "embeddings_initializer", graph, "uniform"
                ),
                embeddings_regularizer=layer_option(
                    params, "embeddings_regularizer", graph, "none"
                ),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                embeddings_constraint=layer_option(params, "embeddings_constraint", graph, "none"),
                mask_zero=params.get("mask_zero", False),
                lora_rank=params.get("lora_rank"),
                lora_alpha=params.get("lora_alpha"),
                quantization_config=params.get("quantization_config"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.conv1d":
            tensors[node_id] = layers.Conv1D(
                params.get("filters", 32),
                params.get("kernel_size", [3]),
                strides=params.get("strides", [1]),
                padding=params.get("padding", "valid"),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1]),
                groups=params.get("groups", 1),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=layer_option(
                    params, "kernel_initializer", graph, "glorot_uniform"
                ),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.conv2d":
            tensors[node_id] = layers.Conv2D(
                params.get("filters", 32),
                params.get("kernel_size", [3, 3]),
                strides=params.get("strides", [1, 1]),
                padding=params.get("padding", "valid"),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1, 1]),
                groups=params.get("groups", 1),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=layer_option(
                    params, "kernel_initializer", graph, "glorot_uniform"
                ),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.conv3d":
            tensors[node_id] = layers.Conv3D(
                params.get("filters", 32),
                params.get("kernel_size", [3, 3, 3]),
                strides=params.get("strides", [1, 1, 1]),
                padding=params.get("padding", "valid"),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1, 1, 1]),
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
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.max_pool2d":
            tensors[node_id] = layers.MaxPooling2D(
                pool_size=params.get("pool_size", [2, 2]),
                strides=params.get("strides"),
                padding=params.get("padding", "valid"),
                data_format=params.get("data_format", "channels_last"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition in {
            "conv.max_pool1d",
            "conv.max_pool3d",
            "conv.average_pool1d",
            "conv.average_pool2d",
            "conv.average_pool3d",
        }:
            class_name = "MaxPooling" if definition.startswith("conv.max") else "AveragePooling"
            dimensions = int(definition[-2])
            pool_type = getattr(layers, f"{class_name}{dimensions}D")
            tensors[node_id] = pool_type(
                pool_size=params.get("pool_size", [2] * dimensions),
                strides=params.get("strides"),
                padding=params.get("padding", "valid"),
                data_format=params.get("data_format", "channels_last"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition in {
            "conv.global_pool1d",
            "conv.global_pool2d",
            "conv.global_pool3d",
        }:
            dimensions = int(definition[-2])
            mode = params.get("mode", "average")
            class_name = "GlobalAveragePooling" if mode == "average" else "GlobalMaxPooling"
            pool_type = getattr(layers, f"{class_name}{dimensions}D")
            tensors[node_id] = pool_type(
                data_format=params.get("data_format", "channels_last"),
                keepdims=params.get("keepdims", False),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.transposed":
            dimensions = int(params.get("dimensions", 2))
            layer_type = getattr(layers, f"Conv{dimensions}DTranspose")
            tensors[node_id] = layer_type(
                params.get("filters", 32),
                params.get("kernel_size", [3] * dimensions),
                strides=params.get("strides", [1] * dimensions),
                padding=params.get("padding", "valid"),
                output_padding=params.get("output_padding"),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1] * dimensions),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                kernel_initializer=layer_option(params, "kernel_initializer", graph, "glorot_uniform"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.depthwise":
            dimensions = int(params.get("dimensions", 2))
            layer_type = getattr(layers, f"DepthwiseConv{dimensions}D")
            tensors[node_id] = layer_type(
                params.get("kernel_size", [3] * dimensions),
                strides=params.get("strides", [1] * dimensions),
                padding=params.get("padding", "valid"),
                depth_multiplier=params.get("depth_multiplier", 1),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1] * dimensions),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                depthwise_initializer=layer_option(params, "depthwise_initializer", graph, "glorot_uniform"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                depthwise_regularizer=layer_option(params, "depthwise_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                depthwise_constraint=layer_option(params, "depthwise_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.separable":
            dimensions = int(params.get("dimensions", 2))
            layer_type = getattr(layers, f"SeparableConv{dimensions}D")
            tensors[node_id] = layer_type(
                params.get("filters", 32),
                params.get("kernel_size", [3] * dimensions),
                strides=params.get("strides", [1] * dimensions),
                padding=params.get("padding", "valid"),
                data_format=params.get("data_format", "channels_last"),
                dilation_rate=params.get("dilation_rate", [1] * dimensions),
                depth_multiplier=params.get("depth_multiplier", 1),
                activation=params.get("activation", "linear"),
                use_bias=params.get("use_bias", True),
                depthwise_initializer=layer_option(params, "depthwise_initializer", graph, "glorot_uniform"),
                pointwise_initializer=layer_option(params, "pointwise_initializer", graph, "glorot_uniform"),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                depthwise_regularizer=layer_option(params, "depthwise_regularizer", graph, "none"),
                pointwise_regularizer=layer_option(params, "pointwise_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                depthwise_constraint=layer_option(params, "depthwise_constraint", graph, "none"),
                pointwise_constraint=layer_option(params, "pointwise_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "conv.upsampling":
            dimensions = int(params.get("dimensions", 2))
            layer_type = getattr(layers, f"UpSampling{dimensions}D")
            kwargs = {**_base_layer_kwargs(params), "name": node.id}
            if dimensions > 1:
                kwargs["data_format"] = params.get("data_format", "channels_last")
            if dimensions == 2:
                kwargs["interpolation"] = params.get("interpolation", "nearest")
            tensors[node_id] = layer_type(params.get("size", [2] * dimensions), **kwargs)(source)
        elif definition in {"conv.padding", "conv.cropping"}:
            dimensions = int(params.get("dimensions", 2))
            prefix = "ZeroPadding" if definition == "conv.padding" else "Cropping"
            layer_type = getattr(layers, f"{prefix}{dimensions}D")
            kwargs = {**_base_layer_kwargs(params), "name": node.id}
            if dimensions > 1:
                kwargs["data_format"] = params.get("data_format", "channels_last")
            key = "padding" if definition == "conv.padding" else "cropping"
            tensors[node_id] = layer_type(params.get(key), **kwargs)(source)
        elif definition in {"recurrent.simple_rnn", "recurrent.lstm", "recurrent.gru"}:
            cell = definition.removeprefix("recurrent.")
            tensors[node_id] = _recurrent_layer(cell, params, graph, name=node.id)(source)
        elif definition == "recurrent.bidirectional":
            inner = _recurrent_layer(
                params.get("cell", "lstm"),
                params,
                graph,
                name=f"{node.id}_forward",
            )
            tensors[node_id] = layers.Bidirectional(
                inner,
                merge_mode=params.get("merge_mode", "concat"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition == "recurrent.stack":
            current = source
            layer_count = int(params.get("layers", 2))
            for index in range(layer_count):
                current = _recurrent_layer(
                    params.get("cell", "lstm"),
                    params,
                    graph,
                    name=f"{node.id}_{index + 1}",
                    return_sequences=(
                        True if index < layer_count - 1 else bool(params.get("return_sequences", False))
                    ),
                )(current)
            tensors[node_id] = current
        elif definition in {"attention.multi_head", "attention.self", "attention.cross"}:
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
                sliding_window=params.get("sliding_window"),
                flash_attention=params.get("flash_attention"),
                use_gate=params.get("use_gate", False),
                kernel_initializer=layer_option(
                    params, "kernel_initializer", graph, "glorot_uniform"
                ),
                bias_initializer=layer_option(params, "bias_initializer", graph, "zeros"),
                kernel_regularizer=layer_option(params, "kernel_regularizer", graph, "none"),
                bias_regularizer=layer_option(params, "bias_regularizer", graph, "none"),
                activity_regularizer=layer_option(params, "activity_regularizer", graph, "none"),
                kernel_constraint=layer_option(params, "kernel_constraint", graph, "none"),
                bias_constraint=layer_option(params, "bias_constraint", graph, "none"),
                seed=params.get("seed"),
                **_base_layer_kwargs(params),
                name=node.id,
            )(query, value)
        elif definition in {"merge.add", "merge.residual", "merge.skip"}:
            tensors[node_id] = layers.Add(name=node.id)(inputs)
        elif definition == "merge.concatenate":
            tensors[node_id] = layers.Concatenate(axis=params.get("axis", -1), name=node.id)(inputs)
        elif definition == "merge.split":
            tensors[node_id] = list(
                keras.ops.split(
                    source,
                    int(params.get("sections", 2)),
                    axis=int(params.get("axis", -1)),
                )
            )
        elif definition == "transformer.positional_encoding":
            source_shape = getattr(source, "shape", None)
            if incoming_edges:
                source_summary = shape_summaries.get(incoming_edges[0].source_node_id)
                if source_summary is not None and source_summary.outputs:
                    source_shape = next(iter(source_summary.outputs.values())).shape
            sequence_length = (
                source_shape[1]
                if source_shape is not None and len(source_shape) > 1
                else None
            )
            feature_dim = (
                source_shape[2]
                if source_shape is not None and len(source_shape) > 2
                else None
            )
            tensors[node_id] = SinusoidalPositionalEncoding(
                sequence_length=(
                    int(sequence_length) if isinstance(sequence_length, int) else None
                ),
                feature_dim=(int(feature_dim) if isinstance(feature_dim, int) else None),
                **_base_layer_kwargs(params),
                name=node.id,
            )(source)
        elif definition in {"transformer.encoder_block", "transformer.stack"}:
            tensors[node_id] = _transformer_encoder_block(source, node.id, params)
        elif definition == "transformer.feed_forward":
            source_dimension = source.shape[-1]
            if source_dimension is None:
                raise RuntimeError("Transformer feed-forward requires a known embedding dimension")
            activation = params.get("activation", "gelu")
            current = layers.Dense(
                int(source_dimension) * int(params.get("ffn_multiplier", 4)),
                activation=None if activation == "gelu" else activation,
                **_base_layer_kwargs(params),
                name=f"{node.id}_expand",
            )(source)
            if activation == "gelu":
                current = GeluActivation(
                    **_base_layer_kwargs(params),
                    name=f"{node.id}_activation",
                )(current)
            current = layers.Dropout(
                params.get("dropout", 0.0),
                **_base_layer_kwargs(params),
                name=f"{node.id}_dropout",
            )(current)
            tensors[node_id] = layers.Dense(
                int(source_dimension),
                **_base_layer_kwargs(params),
                name=f"{node.id}_project",
            )(current)
        elif definition == "transformer.decoder_block":
            query = inputs[0]
            context = inputs[1] if len(inputs) > 1 else query
            source_dimension = query.shape[-1]
            if source_dimension is None:
                raise RuntimeError("Transformer decoder requires a known embedding dimension")
            current = query
            activation = params.get("activation", "gelu")
            hidden_dimension = int(source_dimension) * int(params.get("ffn_multiplier", 4))
            pre_norm = bool(params.get("pre_norm", True))
            base_kwargs = _base_layer_kwargs(params)
            for index in range(int(params.get("layers", 1))):
                suffix = f"_{index + 1}"
                self_norm = layers.LayerNormalization(
                    **base_kwargs, name=f"{node.id}{suffix}_self_norm"
                )
                cross_norm = layers.LayerNormalization(
                    **base_kwargs, name=f"{node.id}{suffix}_cross_norm"
                )
                ffn_norm = layers.LayerNormalization(
                    **base_kwargs, name=f"{node.id}{suffix}_ffn_norm"
                )
                self_input = self_norm(current) if pre_norm else current
                self_attention = layers.MultiHeadAttention(
                    num_heads=params.get("num_heads", 8),
                    key_dim=params.get("key_dim", 64),
                    dropout=params.get("dropout", 0.0),
                    **base_kwargs,
                    name=f"{node.id}{suffix}_self_attention",
                )(self_input, self_input)
                self_residual = layers.Add(name=f"{node.id}{suffix}_self_residual")(
                    [current, self_attention]
                )
                current = self_residual if pre_norm else self_norm(self_residual)
                cross_input = cross_norm(current) if pre_norm else current
                cross_attention = layers.MultiHeadAttention(
                    num_heads=params.get("num_heads", 8),
                    key_dim=params.get("key_dim", 64),
                    dropout=params.get("dropout", 0.0),
                    **base_kwargs,
                    name=f"{node.id}{suffix}_cross_attention",
                )(cross_input, context)
                cross_residual = layers.Add(name=f"{node.id}{suffix}_cross_residual")(
                    [current, cross_attention]
                )
                current = cross_residual if pre_norm else cross_norm(cross_residual)
                feed_forward_input = ffn_norm(current) if pre_norm else current
                feed_forward = layers.Dense(
                    hidden_dimension,
                    activation=None if activation == "gelu" else activation,
                    **base_kwargs,
                    name=f"{node.id}{suffix}_ffn_expand",
                )(feed_forward_input)
                if activation == "gelu":
                    feed_forward = GeluActivation(
                        **base_kwargs,
                        name=f"{node.id}{suffix}_ffn_activation",
                    )(feed_forward)
                feed_forward = layers.Dropout(
                    params.get("dropout", 0.0),
                    **base_kwargs,
                    name=f"{node.id}{suffix}_ffn_dropout",
                )(feed_forward)
                feed_forward = layers.Dense(
                    int(source_dimension),
                    **base_kwargs,
                    name=f"{node.id}{suffix}_ffn_project",
                )(feed_forward)
                ffn_residual = layers.Add(name=f"{node.id}{suffix}_ffn_residual")(
                    [current, feed_forward]
                )
                current = (
                    ffn_residual
                    if pre_norm
                    else layers.LayerNormalization(
                        **base_kwargs, name=f"{node.id}{suffix}_output_norm"
                    )(ffn_residual)
                )
            tensors[node_id] = current
        elif definition == "transformer.residual":
            tensors[node_id] = layers.Add(**_base_layer_kwargs(params), name=node.id)(inputs)
        elif definition == "output.detection":
            tensors[node_id] = tuple(inputs)
        elif definition == "transformer.attention_mask" or manifest.semantic_role == "output_adapter":
            tensors[node_id] = layers.Activation(
                "linear", **_base_layer_kwargs(params), name=node.id
            )(source)
        else:
            raise RuntimeError(f"Keras worker does not support node {definition}")

        output_ports = _enabled_output_ports(manifest, params)
        if not output_ports:
            raise RuntimeError(f"Node {node_id} does not expose an enabled output port")
        result = tensors[node_id]
        if len(output_ports) > 1:
            if not isinstance(result, (tuple, list)) or len(result) != len(output_ports):
                raise RuntimeError(
                    f"Node {node_id} returned {type(result).__name__}, expected "
                    f"{len(output_ports)} output values"
                )
            for port, value in zip(output_ports, result, strict=True):
                port_tensors[(node_id, port.id)] = value
            tensors[node_id] = result[0]
        else:
            port_tensors[(node_id, output_ports[0].id)] = result
        if trace_outputs is not None:
            trace_outputs[node_id] = {
                port.id: port_tensors[(node_id, port.id)] for port in output_ports
            }

    inputs = [tensors[node.id] for node in input_nodes(graph, catalog)]
    outputs = [
        port_tensors[(output.binding.node_id, output.binding.port_id)]
        for output in resolved_outputs(graph, catalog)
    ]
    return keras.Model(inputs=inputs, outputs=outputs, name="neuro_weave_model")
