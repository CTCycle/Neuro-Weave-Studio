from __future__ import annotations

from typing import Any

from domain.compilation import CanonicalModelIR
from domain.graph import ModelGraph
from exporters.base import SourceExporter


###############################################################################
class KerasSourceExporter(SourceExporter):

    # -------------------------------------------------------------------------
    @staticmethod
    def _option(value: object) -> object:
        return None if value in (None, "", "none") else value

    # -------------------------------------------------------------------------
    @classmethod
    def _layer_option(
        cls,
        parameters: dict[str, Any],
        key: str,
        graph: ModelGraph,
        fallback: Any,
    ) -> object:
        if key in parameters:
            return cls._option(parameters[key])
        configuration = graph.model_configuration
        if "initializer" in key:
            return cls._option(configuration.default_initializer)
        if "regularizer" in key:
            return cls._option(configuration.default_regularizer)
        if "constraint" in key:
            return cls._option(configuration.default_constraint)
        return cls._option(fallback)

    # -------------------------------------------------------------------------
    @staticmethod
    def _compile_mapping(
        graph: ModelGraph,
        configuration: dict[str, Any],
        default: Any,
    ) -> Any:
        if not configuration:
            return None
        outputs = [node for node in graph.nodes if node.definition_id.startswith("output.")]
        values: dict[str, Any] = {}
        for node in outputs:
            value = configuration.get("__default__", default)
            if node.id in configuration:
                value = configuration[node.id]
            elif node.label and node.label in configuration:
                value = configuration[node.label]
            values[node.id] = value
        return next(iter(values.values())) if len(values) == 1 else values

    # -------------------------------------------------------------------------
    @staticmethod
    def _optimizer_name(name: str) -> str:
        return {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD", "rmsprop": "RMSprop"}[name]

    # -------------------------------------------------------------------------
    def generate(self, graph: ModelGraph, ir: CanonicalModelIR) -> str:
        nodes = {node.id: node for node in graph.nodes}
        lines = [
            "from __future__ import annotations",
            "",
            "import keras",
            "from keras import layers",
        ]
        if any(node.definition_id == "transformer.positional_encoding" for node in graph.nodes):
            lines.extend(
                [
                    "",
                    "",
                    "@keras.saving.register_keras_serializable(package='NeuroWeave')",
                    "class SinusoidalPositionalEncoding(layers.Layer):",
                    "    def call(self, inputs):",
                    "        length = keras.ops.shape(inputs)[1]",
                    "        depth = keras.ops.shape(inputs)[2]",
                    "        positions = keras.ops.cast(keras.ops.arange(length)[:, None], inputs.dtype)",
                    "        dimensions = keras.ops.cast(keras.ops.arange(depth)[None, :], inputs.dtype)",
                    "        depth_value = keras.ops.cast(depth, inputs.dtype)",
                    "        rates = keras.ops.power(10000.0, -(2.0 * keras.ops.floor(dimensions / 2.0)) / depth_value)",
                    "        angles = positions * rates",
                    "        even = keras.ops.cast(keras.ops.mod(keras.ops.arange(depth), 2) == 0, inputs.dtype)[None, :]",
                    "        encoding = keras.ops.sin(angles) * even + keras.ops.cos(angles) * (1.0 - even)",
                    "        return inputs + encoding[None, :, :]",
                    "",
                    "    def get_config(self):",
                    "        return super().get_config()",
                ]
            )
        lines.extend(["", "", "def build_model() -> keras.Model:"])
        if graph.model_configuration.random_seed is not None:
            lines.append(
                f"    keras.utils.set_random_seed({graph.model_configuration.random_seed!r})"
            )
        policy = "mixed_float16" if graph.model_configuration.mixed_precision else "float32"
        lines.append(f"    keras.mixed_precision.set_global_policy({policy!r})")

        variables: dict[str, str] = {}
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            parameters = node.parameters
            variable = item.variable_name
            variables[item.node_id] = variable
            inputs = [variables[source_id] for source_id in item.input_node_ids]
            source = inputs[0] if inputs else ""
            definition = item.definition_id
            if definition.startswith("input."):
                shape = list(parameters.get("shape", [None, 1]))
                feature_shape = shape[1:] if shape and shape[0] is None else shape
                lines.append(
                    f"    {variable} = keras.Input(shape={feature_shape!r}, name={node.id!r}, dtype={parameters.get('dtype', 'float32')!r})"
                )
            elif definition == "layer.dense":
                lines.append(
                    f"    {variable} = layers.Dense({parameters.get('units', 32)!r}, activation={parameters.get('activation', 'linear')!r}, use_bias={parameters.get('use_bias', True)!r}, kernel_initializer={self._layer_option(parameters, 'kernel_initializer', graph, 'glorot_uniform')!r}, bias_initializer={self._layer_option(parameters, 'bias_initializer', graph, 'zeros')!r}, kernel_regularizer={self._layer_option(parameters, 'kernel_regularizer', graph, 'none')!r}, bias_regularizer={self._layer_option(parameters, 'bias_regularizer', graph, 'none')!r}, activity_regularizer={self._layer_option(parameters, 'activity_regularizer', graph, 'none')!r}, kernel_constraint={self._layer_option(parameters, 'kernel_constraint', graph, 'none')!r}, bias_constraint={self._layer_option(parameters, 'bias_constraint', graph, 'none')!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.activation":
                lines.append(
                    f"    {variable} = layers.Activation({parameters.get('activation', 'relu')!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.dropout":
                lines.append(
                    f"    {variable} = layers.Dropout({parameters.get('rate', 0.5)!r}, seed={parameters.get('seed')!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.flatten":
                lines.append(
                    f"    {variable} = layers.Flatten(data_format={parameters.get('data_format', 'channels_last')!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.batch_norm":
                lines.append(
                    f"    {variable} = layers.BatchNormalization(axis={parameters.get('axis', -1)!r}, momentum={parameters.get('momentum', 0.99)!r}, epsilon={parameters.get('epsilon', 0.001)!r}, center={parameters.get('center', True)!r}, scale={parameters.get('scale', True)!r}, beta_initializer={self._layer_option(parameters, 'beta_initializer', graph, 'zeros')!r}, gamma_initializer={self._layer_option(parameters, 'gamma_initializer', graph, 'ones')!r}, moving_mean_initializer={self._layer_option(parameters, 'moving_mean_initializer', graph, 'zeros')!r}, moving_variance_initializer={self._layer_option(parameters, 'moving_variance_initializer', graph, 'ones')!r}, beta_regularizer={self._layer_option(parameters, 'beta_regularizer', graph, 'none')!r}, gamma_regularizer={self._layer_option(parameters, 'gamma_regularizer', graph, 'none')!r}, beta_constraint={self._layer_option(parameters, 'beta_constraint', graph, 'none')!r}, gamma_constraint={self._layer_option(parameters, 'gamma_constraint', graph, 'none')!r}, synchronized={parameters.get('synchronized', False)!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.layer_norm":
                lines.append(
                    f"    {variable} = layers.LayerNormalization(axis={parameters.get('axis', [-1])!r}, epsilon={parameters.get('epsilon', 0.001)!r}, center={parameters.get('center', True)!r}, scale={parameters.get('scale', True)!r}, beta_initializer={self._layer_option(parameters, 'beta_initializer', graph, 'zeros')!r}, gamma_initializer={self._layer_option(parameters, 'gamma_initializer', graph, 'ones')!r}, beta_regularizer={self._layer_option(parameters, 'beta_regularizer', graph, 'none')!r}, gamma_regularizer={self._layer_option(parameters, 'gamma_regularizer', graph, 'none')!r}, beta_constraint={self._layer_option(parameters, 'beta_constraint', graph, 'none')!r}, gamma_constraint={self._layer_option(parameters, 'gamma_constraint', graph, 'none')!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.embedding":
                lines.append(
                    f"    {variable} = layers.Embedding({parameters.get('vocabulary_size', 10000)!r}, {parameters.get('embedding_dim', 128)!r}, embeddings_initializer={self._layer_option(parameters, 'embeddings_initializer', graph, 'uniform')!r}, embeddings_regularizer={self._layer_option(parameters, 'embeddings_regularizer', graph, 'none')!r}, activity_regularizer={self._layer_option(parameters, 'activity_regularizer', graph, 'none')!r}, embeddings_constraint={self._layer_option(parameters, 'embeddings_constraint', graph, 'none')!r}, mask_zero={parameters.get('mask_zero', False)!r}, lora_rank={parameters.get('lora_rank')!r}, name={node.id!r})({source})"
                )
            elif definition == "conv.conv2d":
                lines.append(
                    f"    {variable} = layers.Conv2D({parameters.get('filters', 32)!r}, {parameters.get('kernel_size', [3, 3])!r}, strides={parameters.get('strides', [1, 1])!r}, padding={parameters.get('padding', 'same')!r}, data_format={parameters.get('data_format', 'channels_last')!r}, dilation_rate={parameters.get('dilation_rate', [1, 1])!r}, groups={parameters.get('groups', 1)!r}, activation={parameters.get('activation', 'linear')!r}, use_bias={parameters.get('use_bias', True)!r}, kernel_initializer={self._layer_option(parameters, 'kernel_initializer', graph, 'glorot_uniform')!r}, bias_initializer={self._layer_option(parameters, 'bias_initializer', graph, 'zeros')!r}, kernel_regularizer={self._layer_option(parameters, 'kernel_regularizer', graph, 'none')!r}, bias_regularizer={self._layer_option(parameters, 'bias_regularizer', graph, 'none')!r}, activity_regularizer={self._layer_option(parameters, 'activity_regularizer', graph, 'none')!r}, kernel_constraint={self._layer_option(parameters, 'kernel_constraint', graph, 'none')!r}, bias_constraint={self._layer_option(parameters, 'bias_constraint', graph, 'none')!r}, name={node.id!r})({source})"
                )
            elif definition == "conv.max_pool2d":
                lines.append(
                    f"    {variable} = layers.MaxPooling2D(pool_size={parameters.get('pool_size', [2, 2])!r}, strides={parameters.get('strides')!r}, padding={parameters.get('padding', 'valid')!r}, data_format={parameters.get('data_format', 'channels_last')!r}, name={node.id!r})({source})"
                )
            elif definition == "recurrent.lstm":
                lines.append(
                    f"    {variable} = layers.LSTM({parameters.get('units', 64)!r}, activation={parameters.get('activation', 'tanh')!r}, recurrent_activation={parameters.get('recurrent_activation', 'sigmoid')!r}, use_bias={parameters.get('use_bias', True)!r}, kernel_initializer={self._layer_option(parameters, 'kernel_initializer', graph, 'glorot_uniform')!r}, recurrent_initializer={self._layer_option(parameters, 'recurrent_initializer', graph, 'orthogonal')!r}, bias_initializer={self._layer_option(parameters, 'bias_initializer', graph, 'zeros')!r}, unit_forget_bias={parameters.get('unit_forget_bias', True)!r}, dropout={parameters.get('dropout', 0.0)!r}, recurrent_dropout={parameters.get('recurrent_dropout', 0.0)!r}, return_sequences={parameters.get('return_sequences', False)!r}, return_state=False, go_backwards={parameters.get('go_backwards', False)!r}, stateful={parameters.get('stateful', False)!r}, unroll={parameters.get('unroll', False)!r}, seed={parameters.get('seed')!r}, name={node.id!r})({source})"
                )
            elif definition == "recurrent.gru":
                lines.append(
                    f"    {variable} = layers.GRU({parameters.get('units', 64)!r}, activation={parameters.get('activation', 'tanh')!r}, recurrent_activation={parameters.get('recurrent_activation', 'sigmoid')!r}, use_bias={parameters.get('use_bias', True)!r}, kernel_initializer={self._layer_option(parameters, 'kernel_initializer', graph, 'glorot_uniform')!r}, recurrent_initializer={self._layer_option(parameters, 'recurrent_initializer', graph, 'orthogonal')!r}, bias_initializer={self._layer_option(parameters, 'bias_initializer', graph, 'zeros')!r}, reset_after={parameters.get('reset_after', True)!r}, dropout={parameters.get('dropout', 0.0)!r}, recurrent_dropout={parameters.get('recurrent_dropout', 0.0)!r}, return_sequences={parameters.get('return_sequences', False)!r}, return_state=False, go_backwards={parameters.get('go_backwards', False)!r}, stateful={parameters.get('stateful', False)!r}, unroll={parameters.get('unroll', False)!r}, seed={parameters.get('seed')!r}, name={node.id!r})({source})"
                )
            elif definition in {"attention.multi_head", "attention.self"}:
                query = inputs[0]
                value = inputs[1] if len(inputs) > 1 else query
                lines.append(
                    f"    {variable} = layers.MultiHeadAttention(num_heads={parameters.get('num_heads', 8)!r}, key_dim={parameters.get('key_dim', 64)!r}, value_dim={parameters.get('value_dim')!r}, dropout={parameters.get('dropout', 0.0)!r}, use_bias={parameters.get('use_bias', True)!r}, output_shape={parameters.get('output_shape')!r}, attention_axes={parameters.get('attention_axes')!r}, kernel_initializer={self._layer_option(parameters, 'kernel_initializer', graph, 'glorot_uniform')!r}, bias_initializer={self._layer_option(parameters, 'bias_initializer', graph, 'zeros')!r}, name={node.id!r})({query}, {value})"
                )
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(
                    f"    {variable} = layers.Add(name={node.id!r})([{', '.join(inputs)}])"
                )
            elif definition == "merge.concatenate":
                lines.append(
                    f"    {variable} = layers.Concatenate(axis={parameters.get('axis', -1)!r}, name={node.id!r})([{', '.join(inputs)}])"
                )
            elif definition == "transformer.positional_encoding":
                lines.append(
                    f"    {variable} = SinusoidalPositionalEncoding(name={node.id!r})({source})"
                )
            elif definition == "transformer.encoder_block":
                layer_count = int(parameters.get("layers", 1))
                multiplier = int(parameters.get("ffn_multiplier", 4))
                activation = parameters.get("activation", "gelu")
                pre_norm = bool(parameters.get("pre_norm", True))
                model_dim = f"{variable}_model_dim"
                hidden_dim = f"{variable}_hidden_dim"
                current = source
                lines.extend(
                    [
                        f"    {model_dim} = int({source}.shape[-1])",
                        f"    {hidden_dim} = {model_dim} * {multiplier!r}",
                    ]
                )
                for index in range(layer_count):
                    suffix = f"_{index + 1}" if layer_count > 1 else ""
                    attention_name = node.id + suffix + "_attention"
                    attention_norm_name = node.id + suffix + "_attention_norm"
                    ffn_norm_name = node.id + suffix + "_ffn_norm"
                    residual_name = node.id + suffix + "_attention_residual"
                    ffn_residual_name = node.id + suffix + "_ffn_residual"
                    if pre_norm:
                        attention_input = f"{variable}_attention_input_{index + 1}"
                        attention = f"{variable}_attention_{index + 1}"
                        attention_residual = f"{variable}_attention_residual_{index + 1}"
                        ffn_input = f"{variable}_ffn_input_{index + 1}"
                        lines.extend(
                            [
                                f"    {attention_input} = layers.LayerNormalization(name={attention_norm_name!r})({current})",
                                f"    {attention} = layers.MultiHeadAttention(num_heads={parameters.get('num_heads', 8)!r}, key_dim={parameters.get('key_dim', 64)!r}, dropout={parameters.get('dropout', 0.0)!r}, name={attention_name!r})({attention_input}, {attention_input})",
                                f"    {attention_residual} = layers.Add(name={residual_name!r})([{current}, {attention}])",
                                f"    {ffn_input} = layers.LayerNormalization(name={ffn_norm_name!r})({attention_residual})",
                            ]
                        )
                    else:
                        attention = f"{variable}_attention_{index + 1}"
                        attention_residual = f"{variable}_attention_residual_{index + 1}"
                        ffn_input = f"{variable}_ffn_input_{index + 1}"
                        lines.extend(
                            [
                                f"    {attention} = layers.MultiHeadAttention(num_heads={parameters.get('num_heads', 8)!r}, key_dim={parameters.get('key_dim', 64)!r}, dropout={parameters.get('dropout', 0.0)!r}, name={attention_name!r})({current}, {current})",
                                f"    {attention_residual} = layers.LayerNormalization(name={attention_norm_name!r})(layers.Add(name={residual_name!r})([{current}, {attention}]))",
                                f"    {ffn_input} = {attention_residual}",
                            ]
                        )
                    feed_forward = f"{variable}_feed_forward_{index + 1}"
                    feed_forward_projected = f"{variable}_feed_forward_projected_{index + 1}"
                    feed_forward_dropout = f"{variable}_feed_forward_dropout_{index + 1}"
                    current = f"{variable}_output_{index + 1}"
                    lines.extend(
                        [
                            f"    {feed_forward} = layers.Dense({hidden_dim}, activation={activation!r}, name={(node.id + suffix + '_ffn_expand')!r})({ffn_input})",
                            f"    {feed_forward_projected} = layers.Dense({model_dim}, name={(node.id + suffix + '_ffn_project')!r})({feed_forward})",
                            f"    {feed_forward_dropout} = layers.Dropout({parameters.get('dropout', 0.0)!r}, name={(node.id + suffix + '_ffn_dropout')!r})({feed_forward_projected})",
                            f"    {current} = layers.Add(name={ffn_residual_name!r})([{attention_residual}, {feed_forward_dropout}])",
                        ]
                    )
                    if not pre_norm:
                        lines.append(
                            f"    {current} = layers.LayerNormalization(name={(node.id + suffix + '_output_norm')!r})({current})"
                        )
                lines.append(f"    {variable} = {current}")
            elif definition.startswith("output."):
                lines.append(
                    f"    {variable} = layers.Activation('linear', name={node.id!r})({source})"
                )
            else:
                raise ValueError(f"No Keras adapter registered for {definition}")

        input_variables = ", ".join(variables[node_id] for node_id in ir.input_node_ids)
        output_variables = ", ".join(variables[node_id] for node_id in ir.output_node_ids)
        lines.append(
            f"    model = keras.Model(inputs=[{input_variables}], outputs=[{output_variables}], name='neuro_weave_model')"
        )
        optimizer_name = self._optimizer_name(graph.model_configuration.optimizer)
        learning_rate = float(
            graph.model_configuration.optimizer_parameters.get("learning_rate", 0.001)
        )
        loss = self._compile_mapping(graph, graph.model_configuration.losses, None)
        metrics = self._compile_mapping(graph, graph.model_configuration.metrics, [])
        lines.extend(
            [
                f"    optimizer = keras.optimizers.{optimizer_name}(learning_rate={learning_rate!r})",
                f"    model.compile(optimizer=optimizer, loss={loss!r}, metrics={metrics!r})",
                "    return model",
                "",
                "",
                "if __name__ == '__main__':",
                "    model = build_model()",
                "    model.summary()",
            ]
        )
        return "\n".join(lines) + "\n"
