from __future__ import annotations

from domain.compilation import CanonicalModelIR
from domain.graph import ModelGraph
from exporters.base import SourceExporter


###############################################################################
class KerasSourceExporter(SourceExporter):

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
                ]
            )
        lines.extend(["", "", "def build_model() -> keras.Model:"])
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
                    f"    {variable} = layers.Dense({parameters.get('units', 32)!r}, activation={parameters.get('activation', 'linear')!r}, use_bias={parameters.get('use_bias', True)!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.activation":
                lines.append(
                    f"    {variable} = layers.Activation({parameters.get('activation', 'relu')!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.dropout":
                lines.append(
                    f"    {variable} = layers.Dropout({parameters.get('rate', 0.5)!r}, name={node.id!r})({source})"
                )
            elif definition == "layer.flatten":
                lines.append(f"    {variable} = layers.Flatten(name={node.id!r})({source})")
            elif definition == "layer.batch_norm":
                lines.append(f"    {variable} = layers.BatchNormalization(name={node.id!r})({source})")
            elif definition == "layer.layer_norm":
                lines.append(f"    {variable} = layers.LayerNormalization(name={node.id!r})({source})")
            elif definition == "layer.embedding":
                lines.append(
                    f"    {variable} = layers.Embedding({parameters.get('vocabulary_size', 10000)!r}, {parameters.get('embedding_dim', 128)!r}, name={node.id!r})({source})"
                )
            elif definition == "conv.conv2d":
                lines.append(
                    f"    {variable} = layers.Conv2D({parameters.get('filters', 32)!r}, {parameters.get('kernel_size', [3, 3])!r}, padding={parameters.get('padding', 'same')!r}, activation={parameters.get('activation', 'linear')!r}, name={node.id!r})({source})"
                )
            elif definition == "conv.max_pool2d":
                lines.append(
                    f"    {variable} = layers.MaxPooling2D(pool_size={parameters.get('pool_size', [2, 2])!r}, name={node.id!r})({source})"
                )
            elif definition == "recurrent.lstm":
                lines.append(
                    f"    {variable} = layers.LSTM({parameters.get('units', 64)!r}, return_sequences={parameters.get('return_sequences', False)!r}, name={node.id!r})({source})"
                )
            elif definition == "recurrent.gru":
                lines.append(
                    f"    {variable} = layers.GRU({parameters.get('units', 64)!r}, return_sequences={parameters.get('return_sequences', False)!r}, name={node.id!r})({source})"
                )
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(
                    f"    {variable} = layers.Add(name={node.id!r})([{', '.join(inputs)}])"
                )
            elif definition == "merge.concatenate":
                lines.append(
                    f"    {variable} = layers.Concatenate(axis={parameters.get('axis', -1)!r}, name={node.id!r})([{', '.join(inputs)}])"
                )
            elif definition in {"attention.multi_head", "attention.self"}:
                query = inputs[0]
                value = inputs[1] if len(inputs) > 1 else query
                lines.append(
                    f"    {variable} = layers.MultiHeadAttention(num_heads={parameters.get('num_heads', 8)!r}, key_dim={parameters.get('key_dim', 64)!r}, name={node.id!r})({query}, {value})"
                )
            elif definition == "transformer.positional_encoding":
                lines.append(
                    f"    {variable} = SinusoidalPositionalEncoding(name={node.id!r})({source})"
                )
            elif definition == "transformer.encoder_block":
                heads = parameters.get("num_heads", 8)
                key_dim = parameters.get("key_dim", 64)
                dropout = parameters.get("dropout", 0.0)
                model_dim = int(heads) * int(key_dim)
                attention_variable = f"{variable}_attention"
                residual_variable = f"{variable}_residual"
                feed_forward_variable = f"{variable}_feed_forward"
                lines.extend(
                    [
                        f"    {attention_variable} = layers.MultiHeadAttention(num_heads={heads!r}, key_dim={key_dim!r}, dropout={dropout!r}, name={node.id + '_attention'!r})({source}, {source})",
                        f"    {residual_variable} = layers.LayerNormalization(name={node.id + '_attention_norm'!r})(layers.Add()([{source}, {attention_variable}]))",
                        f"    {feed_forward_variable} = layers.Dense({model_dim * 4!r}, activation='gelu', name={node.id + '_ffn_expand'!r})({residual_variable})",
                        f"    {feed_forward_variable} = layers.Dense({model_dim!r}, name={node.id + '_ffn_project'!r})({feed_forward_variable})",
                        f"    {feed_forward_variable} = layers.Dropout({dropout!r}, name={node.id + '_ffn_dropout'!r})({feed_forward_variable})",
                        f"    {variable} = layers.LayerNormalization(name={node.id + '_output_norm'!r})(layers.Add()([{residual_variable}, {feed_forward_variable}]))",
                    ]
                )
            elif definition.startswith("output."):
                lines.append(f"    {variable} = {source}")
            else:
                lines.append(f"    {variable} = {source}  # {definition}: identity fallback for supported project serialization only")
        input_variables = ", ".join(variables[node_id] for node_id in ir.input_node_ids)
        output_variables = ", ".join(variables[node_id] for node_id in ir.output_node_ids)
        lines.extend(
            [
                f"    model = keras.Model(inputs=[{input_variables}], outputs=[{output_variables}], name='neuro_weave_model')",
                "    return model",
                "",
                "",
                "if __name__ == '__main__':",
                "    model = build_model()",
                "    model.summary()",
            ]
        )
        return "\n".join(lines) + "\n"
