from __future__ import annotations

from domain.compilation import CanonicalModelIR
from domain.graph import ModelGraph
from exporters.base import SourceExporter


class JaxSourceExporter(SourceExporter):
    """Generate a Flax Linen module for the JAX code target."""

    def generate(self, graph: ModelGraph, ir: CanonicalModelIR) -> str:
        nodes = {node.id: node for node in graph.nodes}
        lines = [
            "from __future__ import annotations",
            "",
            "import jax.numpy as jnp",
            "from flax import linen as nn",
            "",
            "",
            "class GeneratedModel(nn.Module):",
            "    @nn.compact",
            "    def __call__(self, inputs):",
        ]
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
                lines.append(f"        {name} = nn.Dense({int(node.parameters.get('units', 32))}, name={name!r})({source})")
                activation = str(node.parameters.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
            elif definition == "layer.activation":
                lines.append(f"        {name} = {self._activation(str(node.parameters.get('activation', 'relu')))}({source})")
            elif definition == "layer.dropout":
                lines.append(f"        {name} = nn.Dropout(rate={float(node.parameters.get('rate', 0.5))!r}, deterministic=True)({source})")
            elif definition == "layer.flatten":
                lines.append(f"        {name} = jnp.reshape({source}, ({source}.shape[0], -1))")
            elif definition == "layer.embedding":
                lines.append(f"        {name} = nn.Embed(num_embeddings={int(node.parameters.get('vocabulary_size', 10000))}, features={int(node.parameters.get('embedding_dim', 128))}, name={name!r})({source})")
            elif definition == "layer.batch_norm":
                lines.append(
                    f"        {name} = nn.BatchNorm(use_running_average=True, momentum={float(node.parameters.get('momentum', 0.99))!r}, epsilon={float(node.parameters.get('epsilon', 0.001))!r}, name={name!r})({source})"
                )
            elif definition == "layer.layer_norm":
                lines.append(f"        {name} = nn.LayerNorm(epsilon={float(node.parameters.get('epsilon', 0.001))!r}, name={name!r})({source})")
            elif definition == "conv.conv2d":
                lines.append(
                    f"        {name} = nn.Conv(features={int(node.parameters.get('filters', 32))}, kernel_size={tuple(node.parameters.get('kernel_size', [3, 3]))!r}, padding={str(node.parameters.get('padding', 'SAME')).upper()!r}, name={name!r})({source})"
                )
            elif definition == "conv.max_pool2d":
                pool_size = tuple(node.parameters.get("pool_size", [2, 2]))
                lines.append(
                    f"        {name} = nn.max_pool({source}, window_shape={pool_size!r}, strides={pool_size!r}, padding='VALID')"
                )
            elif definition == "recurrent.gru":
                lines.append(
                    f"        {name} = nn.RNN(nn.GRUCell(features={int(node.parameters.get('units', 64))}), return_carry=False, name={name!r})({source})"
                )
            elif definition == "recurrent.lstm":
                lines.append(
                    f"        {name} = nn.RNN(nn.LSTMCell(features={int(node.parameters.get('units', 64))}), return_carry=False, name={name!r})({source})"
                )
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(f"        {name} = sum([{', '.join(source_vars)}])")
            elif definition == "merge.concatenate":
                lines.append(f"        {name} = jnp.concatenate([{', '.join(source_vars)}], axis={int(node.parameters.get('axis', -1))!r})")
            elif definition.startswith("output."):
                lines.append(f"        {name} = {source}")
            else:
                raise ValueError(f"No JAX/Flax adapter registered for {definition}")
            variables[item.node_id] = name
        outputs = [variables[node_id] for node_id in ir.output_node_ids]
        lines.append(f"        return {outputs[0] if len(outputs) == 1 else '(' + ', '.join(outputs) + ')'}")
        lines.extend(["", "", "def build_model() -> GeneratedModel:", "    return GeneratedModel()", ""])
        return "\n".join(lines)

    @staticmethod
    def _activation(value: str) -> str:
        try:
            return {"linear": "lambda value: value", "relu": "nn.relu", "sigmoid": "nn.sigmoid", "tanh": "nn.tanh", "gelu": "nn.gelu", "softmax": "nn.softmax"}[value]
        except KeyError as exc:
            raise ValueError(f"No JAX activation adapter registered for {value}") from exc
