from __future__ import annotations

from domain.compilation import CanonicalModelIR
from domain.graph import ModelGraph
from exporters.base import SourceExporter

###############################################################################
class TensorFlowSourceExporter(SourceExporter):
    """Generate a dependency-light TensorFlow ``tf.Module``."""

    # -------------------------------------------------------------------------
    def generate(self, graph: ModelGraph, ir: CanonicalModelIR) -> str:
        nodes = {node.id: node for node in graph.nodes}
        lines = [
            "from __future__ import annotations",
            "",
            "import tensorflow as tf",
            "",
            "",
            "class GeneratedModel(tf.Module):",
            "    def __init__(self) -> None:",
            "        super().__init__(name='neuro_weave_model')",
        ]
        widths: dict[str, int | None] = {}
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            params = node.parameters
            definition = item.definition_id
            name = item.variable_name
            inputs = [widths.get(source) for source in item.input_node_ids]
            input_width = inputs[0] if inputs else None
            if definition.startswith("input."):
                shape = params.get("shape", [None, 1])
                widths[item.node_id] = shape[-1] if isinstance(shape, list) and isinstance(shape[-1], int) else None
            elif definition == "layer.dense":
                units = int(params.get("units", 32))
                width = input_width or 1
                lines.extend([
                    f"        self.{name}_kernel = tf.Variable(tf.zeros([{width}, {units}], tf.float32), name={name + '_kernel'!r})",
                    f"        self.{name}_bias = tf.Variable(tf.zeros([{units}], tf.float32), name={name + '_bias'!r})",
                ])
                widths[item.node_id] = units
            elif definition == "layer.embedding":
                vocab = int(params.get("vocabulary_size", 10000))
                dim = int(params.get("embedding_dim", 128))
                lines.append(f"        self.{name}_table = tf.Variable(tf.zeros([{vocab}, {dim}], tf.float32), name={name + '_table'!r})")
                widths[item.node_id] = dim
            elif definition == "conv.conv2d":
                lines.append(
                    f"        self.{name} = tf.keras.layers.Conv2D(filters={int(params.get('filters', 32))}, kernel_size={tuple(params.get('kernel_size', [3, 3]))!r}, padding={str(params.get('padding', 'same'))!r}, activation={str(params.get('activation', 'linear'))!r}, name={name!r})"
                )
            elif definition == "conv.max_pool2d":
                lines.append(
                    f"        self.{name} = tf.keras.layers.MaxPool2D(pool_size={tuple(params.get('pool_size', [2, 2]))!r}, padding='valid', name={name!r})"
                )
            elif definition == "layer.batch_norm":
                lines.append(
                    f"        self.{name} = tf.keras.layers.BatchNormalization(momentum={float(params.get('momentum', 0.99))!r}, epsilon={float(params.get('epsilon', 0.001))!r}, name={name!r})"
                )
            elif definition == "layer.layer_norm":
                lines.append(
                    f"        self.{name} = tf.keras.layers.LayerNormalization(epsilon={float(params.get('epsilon', 0.001))!r}, name={name!r})"
                )
            elif definition == "recurrent.gru":
                lines.append(
                    f"        self.{name} = tf.keras.layers.GRU(units={int(params.get('units', 64))}, return_sequences={bool(params.get('return_sequences', False))!r}, name={name!r})"
                )
            elif definition == "recurrent.lstm":
                lines.append(
                    f"        self.{name} = tf.keras.layers.LSTM(units={int(params.get('units', 64))}, return_sequences={bool(params.get('return_sequences', False))!r}, name={name!r})"
                )
                widths[item.node_id] = input_width
            else:
                widths[item.node_id] = input_width
        lines.extend(["", "    @tf.function", "    def __call__(self, inputs: tf.Tensor | dict[str, tf.Tensor]) -> tf.Tensor | tuple[tf.Tensor, ...]:"])
        variables: dict[str, str] = {}
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            definition = item.definition_id
            name = item.variable_name
            source_vars = [variables[source_id] for source_id in item.input_node_ids]
            source = source_vars[0] if source_vars else ""
            if definition.startswith("input."):
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
            elif definition == "layer.dense":
                lines.append(f"        {name} = tf.linalg.matmul({source}, self.{name}_kernel) + self.{name}_bias")
                activation = str(node.parameters.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
            elif definition == "layer.activation":
                lines.append(f"        {name} = {self._activation(str(node.parameters.get('activation', 'relu')))}({source})")
            elif definition == "layer.dropout":
                lines.append(f"        {name} = tf.nn.dropout({source}, rate={float(node.parameters.get('rate', 0.5))!r})")
            elif definition == "layer.flatten":
                lines.append(f"        {name} = tf.reshape({source}, [tf.shape({source})[0], -1])")
            elif definition == "layer.embedding":
                lines.append(f"        {name} = tf.gather(self.{name}_table, tf.cast({source}, tf.int32))")
            elif definition in {"layer.batch_norm", "layer.layer_norm", "conv.conv2d", "conv.max_pool2d", "recurrent.gru", "recurrent.lstm"}:
                lines.append(f"        {name} = self.{name}({source})")
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(f"        {name} = tf.add_n([{', '.join(source_vars)}])")
            elif definition == "merge.concatenate":
                lines.append(f"        {name} = tf.concat([{', '.join(source_vars)}], axis={int(node.parameters.get('axis', -1))!r})")
            elif definition.startswith("output."):
                lines.append(f"        {name} = {source}")
            else:
                raise ValueError(f"No TensorFlow adapter registered for {definition}")
            variables[item.node_id] = name
        outputs = [variables[node_id] for node_id in ir.output_node_ids]
        lines.append(f"        return {outputs[0] if len(outputs) == 1 else '(' + ', '.join(outputs) + ')'}")
        lines.extend(["", "", "def build_model() -> GeneratedModel:", "    return GeneratedModel()", ""])
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    @staticmethod
    def _activation(value: str) -> str:
        try:
            return {"relu": "tf.nn.relu", "sigmoid": "tf.nn.sigmoid", "tanh": "tf.nn.tanh", "gelu": "tf.nn.gelu", "softmax": "tf.nn.softmax", "linear": "tf.identity"}[value]
        except KeyError as exc:
            raise ValueError(f"No TensorFlow activation adapter registered for {value}") from exc
