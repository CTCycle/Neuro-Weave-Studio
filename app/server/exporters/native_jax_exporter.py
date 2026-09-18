# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import CompilationCatalog
from domain.graph import ModelGraph
from exporters.base import SourceExporter


###############################################################################
class NativeJaxSourceExporter(SourceExporter):
    """Generate a native Flax Linen module for the verified subset."""

    # -------------------------------------------------------------------------
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
    @staticmethod
    def _condition_matches(condition: dict[str, object], parameters: dict[str, object]) -> bool:
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

    # -------------------------------------------------------------------------
    def generate(
        self,
        graph: ModelGraph,
        ir: CanonicalModelIR,
        catalog: NodeCatalog | None = None,
        *,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> str:
        del compilation_catalog
        catalog = catalog or get_node_catalog()
        nodes = {node.id: node for node in graph.nodes}
        summaries = {
            summary.node_id: next(iter(summary.outputs.values()))
            for summary in ir.shape_summaries
            if summary.outputs
        }
        lines = [
            "from __future__ import annotations",
            "",
            "import jax",
            "import jax.numpy as jnp",
            "from flax import linen as nn",
            "",
            "",
            "class GeneratedModel(nn.Module):",
            "    @nn.compact",
            "    def __call__(self, inputs, return_intermediates=False):",
        ]
        variables: dict[str, str] = {}
        port_variables: dict[tuple[str, str], str] = {}
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            definition = item.definition_id
            name = item.variable_name
            params = node.parameters
            source_vars = [
                port_variables[(binding.source_node_id, binding.source_port_id)]
                for binding in item.input_bindings
            ]
            source = source_vars[0] if source_vars else ""
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None
            manifest = catalog.get(definition, node.definition_version)
            if definition in {"input.constant", "input.parameter"}:
                if not ir.input_node_ids:
                    raise ValueError(
                        f"Generated source node {node.id} requires at least one runtime model input for batch materialization"
                    )
                anchor = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{ir.input_node_ids[0]!r}]"
                source_shape = params.get("shape", [1])
                if not isinstance(source_shape, list):
                    raise ValueError(f"Generated source node {node.id} requires a list shape")
                dtype = str(params.get("dtype", "float32"))
                if definition == "input.constant":
                    lines.append(
                        f"        {name}_source = jnp.full({tuple(source_shape)!r}, {params.get('value', 0)!r}, dtype=jnp.{dtype})"
                    )
                else:
                    initializer = str(params.get("initializer", "zeros"))
                    initializer_source = {
                        "zeros": "nn.initializers.zeros_init()",
                        "ones": "nn.initializers.ones_init()",
                        "random_normal": "nn.initializers.normal()",
                    }.get(initializer)
                    if initializer_source is None:
                        raise ValueError(f"Unsupported JAX parameter initializer: {initializer}")
                    lines.append(
                        f"        {name}_source = self.param({name + '_parameter'!r}, {initializer_source}, {tuple(source_shape)!r}, jnp.{dtype})"
                    )
                lines.append(
                    f"        {name} = jnp.repeat(jnp.expand_dims({name}_source, 0), {anchor}.shape[0], axis=0)"
                )
            elif manifest.operation is not None:
                lines.append(
                    f"        {name} = {emit_operation_source(manifest, 'jax', source_vars, params)}"
                )
            elif manifest.semantic_role == "input":
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
            elif definition == "layer.reshape":
                target_shape = tuple(params.get("target_shape", [-1]))
                lines.append(f"        {name} = jnp.reshape({source}, ({source}.shape[0], *{target_shape!r}))")
            elif definition == "layer.permute":
                axes = tuple(int(value) for value in params.get("axes", [2, 1]))
                lines.append(f"        {name} = jnp.transpose({source}, {(0, *axes)!r})")
            elif definition == "layer.repeat":
                repeats = int(params.get("repeats", 2))
                axis = int(params.get("axis", 1))
                if axis == 1 and source_spec is not None and source_spec.rank == 2:
                    lines.append(
                        f"        {name} = jnp.repeat(jnp.expand_dims({source}, axis=1), {repeats!r}, axis=1)"
                    )
                else:
                    lines.append(
                        f"        {name} = jnp.repeat({source}, {repeats!r}, axis={axis!r})"
                    )
            elif definition == "layer.unit_norm":
                axis = int(params.get("axis", -1))
                lines.append(
                    f"        {name}_norm = jnp.sqrt(jnp.sum(jnp.square({source}), axis={axis!r}, keepdims=True))"
                )
                lines.append(
                    f"        {name} = {source} / jnp.maximum({name}_norm, jnp.asarray(1e-12, dtype={source}.dtype))"
                )
            elif definition == "layer.rms_norm":
                axis = int(params.get("axis", -1))
                epsilon = float(params.get("epsilon", 1e-6))
                lines.append(
                    f"        {name}_gamma = self.param({name + '_gamma'!r}, nn.initializers.ones_init(), ({source}.shape[{axis!r}],), {source}.dtype)"
                )
                lines.append(
                    f"        {name} = {source} * jax.lax.rsqrt(jnp.mean(jnp.square({source}), axis={axis!r}, keepdims=True) + {epsilon!r})"
                )
                lines.append(f"        {name}_shape = [1] * {source}.ndim")
                lines.append(f"        {name}_shape[{axis!r}] = {name}_gamma.shape[0]")
                lines.append(f"        {name} = {name} * jnp.reshape({name}_gamma, {name}_shape)")
            elif definition == "layer.prelu":
                shared_axes = params.get("shared_axes")
                shared_axes_values = tuple(shared_axes) if isinstance(shared_axes, list) else ()
                lines.append(
                    f"        {name}_alpha_shape = tuple(1 if axis in {shared_axes_values!r} else {source}.shape[axis] for axis in range(1, {source}.ndim))"
                )
                lines.append(
                    f"        {name}_alpha = self.param({name + '_alpha'!r}, nn.initializers.zeros_init(), {name}_alpha_shape, {source}.dtype)"
                )
                lines.append(
                    f"        {name} = jnp.where({source} >= 0, {source}, jnp.reshape({name}_alpha, (1, *{name}_alpha.shape)) * {source})"
                )
            elif definition == "layer.group_norm":
                axis = int(params.get("axis", -1))
                groups = int(params.get("groups", 32))
                epsilon = float(params.get("epsilon", 0.001))
                lines.append(f"        {name}_axis = {axis!r} if {axis!r} >= 0 else {source}.ndim + {axis!r}")
                lines.append(
                    f"        {name}_permutation = tuple(index for index in range({source}.ndim) if index != {name}_axis) + ({name}_axis,)"
                )
                lines.append(
                    f"        {name}_inverse = tuple({name}_permutation.index(index) for index in range({source}.ndim))"
                )
                lines.append(f"        {name}_input = jnp.transpose({source}, {name}_permutation)")
                lines.append(f"        {name}_channels = {name}_input.shape[-1]")
                lines.append(
                    f"        {name}_grouped = jnp.reshape({name}_input, (*{name}_input.shape[:-1], {groups!r}, {name}_channels // {groups!r}))"
                )
                lines.append(
                    f"        {name}_axes = tuple(range(1, {name}_grouped.ndim - 2)) + ({name}_grouped.ndim - 1,)"
                )
                lines.append(
                    f"        {name}_mean = jnp.mean({name}_grouped, axis={name}_axes, keepdims=True)"
                )
                lines.append(
                    f"        {name}_variance = jnp.var({name}_grouped, axis={name}_axes, keepdims=True)"
                )
                lines.append(
                    f"        {name} = jnp.reshape(({name}_grouped - {name}_mean) * jax.lax.rsqrt({name}_variance + {epsilon!r}), {name}_input.shape)"
                )
                if bool(params.get("scale", True)):
                    lines.append(
                        f"        {name}_gamma = self.param({name + '_gamma'!r}, nn.initializers.ones_init(), ({name}_channels,), {source}.dtype)"
                    )
                    lines.append(f"        {name} = {name} * {name}_gamma")
                if bool(params.get("center", True)):
                    lines.append(
                        f"        {name}_beta = self.param({name + '_beta'!r}, nn.initializers.zeros_init(), ({name}_channels,), {source}.dtype)"
                    )
                    lines.append(f"        {name} = {name} + {name}_beta")
                lines.append(f"        {name} = jnp.transpose({name}, {name}_inverse)")
            elif definition == "layer.dense":
                lines.append(
                    f"        {name} = nn.Dense({int(params.get('units', 32))}, use_bias={bool(params.get('use_bias', True))!r}, name={name!r})({source})"
                )
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
            elif definition == "layer.activation":
                lines.append(
                    f"        {name} = {self._activation(str(params.get('activation', 'relu')))}({source})"
                )
            elif definition == "layer.dropout":
                lines.append(
                    f"        {name} = nn.Dropout(rate={float(params.get('rate', 0.5))!r}, deterministic=True, name={name!r})({source})"
                )
            elif definition == "layer.flatten":
                lines.append(f"        {name} = jnp.reshape({source}, ({source}.shape[0], -1))")
            elif definition == "layer.embedding":
                lines.append(
                    f"        {name} = nn.Embed(num_embeddings={int(params.get('vocabulary_size', 10000))}, features={int(params.get('embedding_dim', 128))}, name={name!r})({source}.astype(jnp.int32))"
                )
            elif definition == "layer.batch_norm":
                axis = params.get("axis", -1)
                if axis not in {-1, 0}:
                    raise ValueError("Native JAX BatchNorm supports the trailing axis only")
                lines.append(
                    f"        {name} = nn.BatchNorm(use_running_average=True, use_scale={bool(params.get('scale', True))!r}, use_bias={bool(params.get('center', True))!r}, momentum={float(params.get('momentum', 0.99))!r}, epsilon={float(params.get('epsilon', 0.001))!r}, axis={axis!r}, name={name!r})({source})"
                )
            elif definition == "layer.layer_norm":
                axes = params.get("axis", [-1])
                if not isinstance(axes, list) or axes != [-1]:
                    raise ValueError("Native JAX LayerNorm supports the trailing axis only")
                lines.append(
                    f"        {name} = nn.LayerNorm(use_scale={bool(params.get('scale', True))!r}, use_bias={bool(params.get('center', True))!r}, epsilon={float(params.get('epsilon', 0.001))!r}, name={name!r})({source})"
                )
            elif definition == "conv.conv1d":
                data_format = str(params.get("data_format", "channels_last"))
                if data_format == "channels_first":
                    lines.append(f"        {name}_input = {source}.transpose(0, 2, 1)")
                else:
                    lines.append(f"        {name}_input = {source}")
                kernel_size = params.get("kernel_size", [3])
                strides = params.get("strides", [1])
                dilation = params.get("dilation_rate", [1])
                if not isinstance(kernel_size, list) or len(kernel_size) != 1:
                    raise ValueError("Native JAX Conv1D expects one kernel_size value")
                if not isinstance(strides, list) or len(strides) != 1:
                    raise ValueError("Native JAX Conv1D expects one strides value")
                if not isinstance(dilation, list) or len(dilation) != 1:
                    raise ValueError("Native JAX Conv1D expects one dilation_rate value")
                padding = str(params.get("padding", "same")).upper()
                lines.append(
                    f"        {name} = nn.Conv(features={int(params.get('filters', 32))}, kernel_size={(int(kernel_size[0]),)!r}, strides={(int(strides[0]),)!r}, kernel_dilation={(int(dilation[0]),)!r}, padding={padding!r}, feature_group_count={int(params.get('groups', 1))}, use_bias={bool(params.get('use_bias', True))!r}, name={name!r})({name}_input)"
                )
                if data_format == "channels_first":
                    lines.append(f"        {name} = {name}.transpose(0, 2, 1)")
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
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
                lines.append(
                    f"        {name} = nn.Conv(features={int(params.get('filters', 32))}, kernel_size={kernel_size!r}, strides={strides!r}, input_dilation={dilation!r}, padding={padding!r}, use_bias={bool(params.get('use_bias', True))!r}, name={name!r})({name}_input)"
                )
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
                lines.append(
                    f"        {name} = nn.max_pool({name}_input, window_shape={pool_size!r}, strides={strides!r}, padding={padding!r})"
                )
                if data_format == "channels_first":
                    lines.append(f"        {name} = {name}.transpose(0, 3, 1, 2)")
            elif definition == "recurrent.lstm":
                if (
                    params.get("activation", "tanh") != "tanh"
                    or params.get("recurrent_activation", "sigmoid") != "sigmoid"
                ):
                    raise ValueError("Native JAX recurrent export supports tanh/sigmoid only")
                lines.append(
                    f"        {name} = nn.RNN(nn.LSTMCell(features={int(params.get('units', 64))}), return_carry=False, name={name!r})({source})"
                )
                if not bool(params.get("return_sequences", False)):
                    lines.append(f"        {name} = {name}[:, -1, :]")
            elif definition == "recurrent.gru":
                if (
                    params.get("activation", "tanh") != "tanh"
                    or params.get("recurrent_activation", "sigmoid") != "sigmoid"
                ):
                    raise ValueError("Native JAX recurrent export supports tanh/sigmoid only")
                lines.append(
                    f"        {name} = nn.RNN(nn.GRUCell(features={int(params.get('units', 64))}), return_carry=False, name={name!r})({source})"
                )
                if not bool(params.get("return_sequences", False)):
                    lines.append(f"        {name} = {name}[:, -1, :]")
            elif definition in {"merge.add", "merge.residual", "merge.skip", "transformer.residual"}:
                lines.append(f"        {name} = sum([{', '.join(source_vars)}])")
            elif definition == "merge.concatenate":
                lines.append(
                    f"        {name} = jnp.concatenate([{', '.join(source_vars)}], axis={int(params.get('axis', -1))!r})"
                )
            elif definition == "merge.split":
                lines.append(
                    f"        {name} = list(jnp.split({source}, {int(params.get('sections', 2))!r}, axis={int(params.get('axis', -1))!r}))"
                )
            elif definition == "transformer.attention_mask":
                lines.append(f"        {name} = {source}")
            elif definition == "output.detection":
                lines.append(f"        {name} = ({', '.join(source_vars)})")
            elif manifest.semantic_role == "output_adapter":
                lines.append(f"        {name} = {source}")
            else:
                raise ValueError(f"No native JAX adapter registered for {definition}")
            output_ports = [
                port
                for port in manifest.outputs
                if self._condition_matches(port.enabled_when, params)
            ]
            if len(output_ports) > 1:
                for index, port in enumerate(output_ports):
                    port_variables[(item.node_id, port.id)] = f"{name}[{index}]"
                variables[item.node_id] = f"{name}[0]"
            else:
                port_variables[(item.node_id, output_ports[0].id)] = name
                variables[item.node_id] = name
        outputs = [port_variables[(binding.node_id, binding.port_id)] for binding in ir.output_bindings]
        result = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
        trace = "{" + ", ".join(
            f"{item.node_id!r}: {{'output': {variables[item.node_id]}}}"
            for item in ir.ordered_nodes
        ) + "}"
        lines.append(f"        if return_intermediates: return {result}, {trace}")
        lines.append(f"        return {result}")
        jit_enabled = graph.model_configuration.native.jax.jit_enabled
        lines.extend(
            [
                "",
                "",
                "def build_model() -> GeneratedModel:",
                "    return GeneratedModel()",
                "",
                "",
                "def build_apply(model: GeneratedModel):",
                f"    if {jit_enabled!r}:",
                "        import jax",
                "        return jax.jit(model.apply)",
                "    return model.apply",
                "",
            ]
        )
        return "\n".join(lines)
