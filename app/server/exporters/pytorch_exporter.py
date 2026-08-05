from __future__ import annotations

from domain.compilation import CanonicalModelIR
from domain.graph import ModelGraph
from exporters.base import SourceExporter


###############################################################################
class PyTorchSourceExporter(SourceExporter):

    # -------------------------------------------------------------------------
    def generate(self, graph: ModelGraph, ir: CanonicalModelIR) -> str:
        nodes = {node.id: node for node in graph.nodes}
        lines = [
            "from __future__ import annotations",
            "",
            "import torch",
            "from torch import Tensor, nn",
            "",
            "",
            "class GeneratedModel(nn.Module):",
            "    def __init__(self) -> None:",
            "        super().__init__()",
        ]
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            params = node.parameters
            definition = item.definition_id
            name = item.variable_name
            if definition == "layer.dense":
                input_features = params.get("input_features")
                if not isinstance(input_features, int):
                    lines.append(
                        f"        self.{name} = nn.LazyLinear({params.get('units', 32)!r})"
                    )
                else:
                    lines.append(
                        f"        self.{name} = nn.Linear({input_features!r}, {params.get('units', 32)!r}, bias={params.get('use_bias', True)!r})"
                    )
            elif definition == "layer.dropout":
                lines.append(f"        self.{name} = nn.Dropout({params.get('rate', 0.5)!r})")
            elif definition == "layer.flatten":
                lines.append(f"        self.{name} = nn.Flatten(start_dim=1)")
            elif definition == "layer.batch_norm":
                lines.append(f"        self.{name} = nn.LazyBatchNorm1d()")
            elif definition == "layer.layer_norm":
                normalized_shape = params.get("normalized_shape", params.get("features", 1))
                lines.append(f"        self.{name} = nn.LayerNorm({normalized_shape!r})")
            elif definition == "layer.embedding":
                lines.append(
                    f"        self.{name} = nn.Embedding({params.get('vocabulary_size', 10000)!r}, {params.get('embedding_dim', 128)!r})"
                )
            elif definition == "conv.conv2d":
                lines.append(
                    f"        self.{name} = nn.LazyConv2d({params.get('filters', 32)!r}, kernel_size={params.get('kernel_size', [3, 3])!r}, padding='same')"
                )
            elif definition == "conv.max_pool2d":
                lines.append(
                    f"        self.{name} = nn.MaxPool2d(kernel_size={params.get('pool_size', [2, 2])!r})"
                )
            elif definition == "recurrent.lstm":
                lines.append(
                    f"        self.{name} = nn.LSTM(input_size={params.get('input_size', 1)!r}, hidden_size={params.get('units', 64)!r}, batch_first=True)"
                )
            elif definition == "recurrent.gru":
                lines.append(
                    f"        self.{name} = nn.GRU(input_size={params.get('input_size', 1)!r}, hidden_size={params.get('units', 64)!r}, batch_first=True)"
                )
        lines.extend(["", "    def forward(self, inputs: Tensor | dict[str, Tensor]) -> Tensor | tuple[Tensor, ...]:"])
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
                variables[item.node_id] = name
            elif definition in {"layer.dense", "layer.dropout", "layer.flatten", "layer.batch_norm", "layer.layer_norm", "layer.embedding", "conv.conv2d", "conv.max_pool2d"}:
                lines.append(f"        {name} = self.{name}({source})")
                variables[item.node_id] = name
            elif definition == "layer.activation":
                activation = node.parameters.get("activation", "relu")
                function = {"relu": "torch.relu", "sigmoid": "torch.sigmoid", "tanh": "torch.tanh", "softmax": "lambda value: torch.softmax(value, dim=-1)"}.get(str(activation), "lambda value: value")
                lines.append(f"        {name} = ({function})({source})")
                variables[item.node_id] = name
            elif definition in {"recurrent.lstm", "recurrent.gru"}:
                lines.append(f"        {name}, _ = self.{name}({source})")
                if not node.parameters.get("return_sequences", False):
                    lines.append(f"        {name} = {name}[:, -1, :]")
                variables[item.node_id] = name
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(f"        {name} = {' + '.join(source_vars)}")
                variables[item.node_id] = name
            elif definition == "merge.concatenate":
                lines.append(f"        {name} = torch.cat([{', '.join(source_vars)}], dim={node.parameters.get('axis', -1)!r})")
                variables[item.node_id] = name
            elif definition.startswith("output."):
                lines.append(f"        {name} = {source}")
                variables[item.node_id] = name
            else:
                lines.append(f"        {name} = {source}  # {definition}: unsupported operation kept as identity")
                variables[item.node_id] = name
        outputs = [variables[node_id] for node_id in ir.output_node_ids]
        lines.append(f"        return {outputs[0] if len(outputs) == 1 else '(' + ', '.join(outputs) + ')'}")
        return "\n".join(lines) + "\n"
