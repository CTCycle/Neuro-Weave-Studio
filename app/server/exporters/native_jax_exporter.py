from __future__ import annotations

from domain.compilation import CanonicalModelIR
from domain.graph import ModelGraph
from exporters.base import SourceExporter


###############################################################################
class NativeJaxSourceExporter(SourceExporter):
    """Generate a native Flax Linen module for the verified subset."""

    @staticmethod
    def _activation(value: str) -> str:
        try:
            return {
                "linear": "lambda value: value",
                "relu": "nn.relu",
                "sigmoid": "nn.sigmoid",
                "tanh": "nn.tanh",
                "gelu": "nn.gelu",
                "softmax": "(lambda value: nn.softmax(value, axis=-1))",
            }[value]
        except KeyError as exc:
            raise ValueError(f"No native JAX activation adapter registered for {value}") from exc

    # -------------------------------------------------------------------------
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
            params = node.parameters
            source_vars = [variables[source_id] for source_id in item.input_node_ids]
            source = source_vars[0] if source_vars else ""
            if definition.startswith("input."):
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
            elif definition == "layer.dense":
                lines.append(f"        {name} = nn.Dense({int(params.get('units', 32))}, use_bias={bool(params.get('use_bias', True))!r}, name={name!r})({source})")
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
            elif definition == "layer.activation":
                lines.append(f"        {name} = {self._activation(str(params.get('activation', 'relu')))}({source})")
            elif definition == "layer.dropout":
                lines.append(f"        {name} = nn.Dropout(rate={float(params.get('rate', 0.5))!r}, deterministic=True, name={name!r})({source})")
            elif definition == "layer.flatten":
                lines.append(f"        {name} = jnp.reshape({source}, ({source}.shape[0], -1))")
            elif definition == "layer.embedding":
                lines.append(f"        {name} = nn.Embed(num_embeddings={int(params.get('vocabulary_size', 10000))}, features={int(params.get('embedding_dim', 128))}, name={name!r})({source}.astype(jnp.int32))")
            elif definition == "layer.batch_norm":
                axis = params.get("axis", -1)
                if axis not in {-1, 0}:
                    raise ValueError("Native JAX BatchNorm supports the trailing axis only")
                lines.append(f"        {name} = nn.BatchNorm(use_running_average=True, use_scale={bool(params.get('scale', True))!r}, use_bias={bool(params.get('center', True))!r}, momentum={float(params.get('momentum', 0.99))!r}, epsilon={float(params.get('epsilon', 0.001))!r}, axis={axis!r}, name={name!r})({source})")
            elif definition == "layer.layer_norm":
                axes = params.get("axis", [-1])
                if not isinstance(axes, list) or axes != [-1]:
                    raise ValueError("Native JAX LayerNorm supports the trailing axis only")
                lines.append(f"        {name} = nn.LayerNorm(use_scale={bool(params.get('scale', True))!r}, use_bias={bool(params.get('center', True))!r}, epsilon={float(params.get('epsilon', 0.001))!r}, name={name!r})({source})")
            elif definition == "conv.conv2d":
                data_format = str(params.get("data_format", "channels_last"))
                if data_format == "channels_first":
                    lines.append(f"        {name}_input = {source}.transpose(0, 2, 3, 1)")
                else:
                    lines.append(f"        {name}_input = {source}")
                padding = str(params.get("padding", "same")).upper()
                kernel_size = tuple(params.get("kernel_size", [3, 3]))
                strides = tuple(params.get("strides", [1, 1]))
                dilation = tuple(params.get("dilation_rate", [1, 1]))
                if int(params.get("groups", 1)) != 1:
                    raise ValueError("Native JAX Conv2D currently supports groups=1 only")
                lines.append(f"        {name} = nn.Conv(features={int(params.get('filters', 32))}, kernel_size={kernel_size!r}, strides={strides!r}, input_dilation={dilation!r}, padding={padding!r}, use_bias={bool(params.get('use_bias', True))!r}, name={name!r})({name}_input)")
                if data_format == "channels_first":
                    lines.append(f"        {name} = {name}.transpose(0, 3, 1, 2)")
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
            elif definition == "conv.max_pool2d":
                data_format = str(params.get("data_format", "channels_last"))
                if data_format == "channels_first":
                    lines.append(f"        {name}_input = {source}.transpose(0, 2, 3, 1)")
                else:
                    lines.append(f"        {name}_input = {source}")
                pool_size = tuple(params.get("pool_size", [2, 2]))
                strides_value = params.get("strides")
                strides = pool_size if strides_value is None else tuple(strides_value)
                padding = str(params.get("padding", "valid")).upper()
                lines.append(f"        {name} = nn.max_pool({name}_input, window_shape={pool_size!r}, strides={strides!r}, padding={padding!r})")
                if data_format == "channels_first":
                    lines.append(f"        {name} = {name}.transpose(0, 3, 1, 2)")
            elif definition == "recurrent.lstm":
                if params.get("activation", "tanh") != "tanh" or params.get("recurrent_activation", "sigmoid") != "sigmoid":
                    raise ValueError("Native JAX recurrent export supports tanh/sigmoid only")
                lines.append(f"        {name} = nn.RNN(nn.LSTMCell(features={int(params.get('units', 64))}), return_carry=False, name={name!r})({source})")
                if not bool(params.get("return_sequences", False)):
                    lines.append(f"        {name} = {name}[:, -1, :]")
            elif definition == "recurrent.gru":
                if params.get("activation", "tanh") != "tanh" or params.get("recurrent_activation", "sigmoid") != "sigmoid":
                    raise ValueError("Native JAX recurrent export supports tanh/sigmoid only")
                lines.append(f"        {name} = nn.RNN(nn.GRUCell(features={int(params.get('units', 64))}), return_carry=False, name={name!r})({source})")
                if not bool(params.get("return_sequences", False)):
                    lines.append(f"        {name} = {name}[:, -1, :]")
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(f"        {name} = sum([{', '.join(source_vars)}])")
            elif definition == "merge.concatenate":
                lines.append(f"        {name} = jnp.concatenate([{', '.join(source_vars)}], axis={int(params.get('axis', -1))!r})")
            elif definition.startswith("output."):
                lines.append(f"        {name} = {source}")
            else:
                raise ValueError(f"No native JAX adapter registered for {definition}")
            variables[item.node_id] = name
        outputs = [variables[node_id] for node_id in ir.output_node_ids]
        lines.append(f"        return {outputs[0] if len(outputs) == 1 else '(' + ', '.join(outputs) + ')'}")
        lines.extend(["", "", "def build_model() -> GeneratedModel:", "    return GeneratedModel()", ""])
        return "\n".join(lines)
