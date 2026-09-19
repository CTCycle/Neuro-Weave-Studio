# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import CompilationCatalog
from domain.graph import ModelGraph
from domain.ports import materialized_outputs
from exporters.base import SourceExporter
from validation.parameter_values import condition_matches
from validation.safe_expression import normalize_index_expression, validate_safe_expression


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
    def _index_expression(value: object) -> str:
        expression = normalize_index_expression(value if isinstance(value, str) else ":")
        validate_safe_expression(expression)
        return expression

    # -------------------------------------------------------------------------
    @staticmethod
    def _initializer_expression(value: object, default: str) -> str:
        identifier = value.get("id", default) if isinstance(value, dict) else value
        init_name = {
            "ones": "ones",
            "random_normal": "normal",
            "normal": "normal",
            "random_uniform": "uniform",
            "uniform": "uniform",
            "zeros": "zeros",
        }.get(identifier if isinstance(identifier, str) else default, default)
        return (
            f"nn.initializers.{init_name}"
            if init_name in {"ones", "zeros"}
            else f"nn.initializers.{init_name}()"
        )

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
            manifest = catalog.get(definition, node.definition_version)
            if manifest.operation is not None:
                lines.append(
                    f"        {name} = {emit_operation_source(manifest, 'jax', source_vars, params)}"
                )
            elif manifest.source_kind == "runtime_input":
                expression = "inputs" if len(ir.input_node_ids) == 1 else f"inputs[{node.id!r}]"
                lines.append(f"        {name} = {expression}")
            elif definition == "input.constant":
                lines.append(
                    f"        {name} = jnp.broadcast_to(jnp.asarray({params.get('value', 0)!r}, dtype=jnp.dtype({params.get('dtype', 'float32')!r})), {tuple(params.get('shape', [1]))!r})"
                )
            elif definition == "input.parameter":
                shape = tuple(params.get("shape", [1]))
                dtype = params.get("dtype", "float32")
                initializer = params.get("initializer", "zeros")
                initializer_expression = self._initializer_expression(initializer, "zeros")
                if bool(params.get("trainable", True)):
                    lines.append(
                        f"        {name} = self.param({name!r}, {initializer_expression}, {shape!r}, jnp.dtype({dtype!r}))"
                    )
                else:
                    lines.append(
                        f"        {name} = self.variable('constants', {name!r}, lambda: jnp.zeros({shape!r}, dtype=jnp.dtype({dtype!r}))).value"
                    )
            elif definition == "math.index":
                expression = self._index_expression(params.get("expression", ":"))
                lines.append(f"        {name} = {expression.replace('x', source, 1)}")
            elif definition == "merge.average":
                lines.append(f"        {name} = jnp.stack([{', '.join(source_vars)}], axis=0).sum(axis=0) / {len(source_vars)!r}")
            elif definition == "merge.split":
                lines.append(
                    f"        {name} = jnp.split({source}, {int(params.get('sections', 2))!r}, axis={int(params.get('axis', -1))!r})"
                )
            elif definition == "output.detection":
                lines.append(f"        {name} = ({', '.join(source_vars)})")
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
            elif definition == "layer.reshape":
                target_shape = tuple(int(value) for value in params.get("target_shape", [-1]))
                lines.append(
                    f"        {name} = jnp.reshape({source}, ({source}.shape[0],) + {target_shape!r})"
                )
            elif definition == "layer.permute":
                axes = tuple(int(value) for value in params.get("axes", [2, 1]))
                lines.append(f"        {name} = jnp.transpose({source}, (0,) + {axes!r})")
            elif definition == "layer.repeat":
                repeats = int(params.get("repeats", 2))
                axis = int(params.get("axis", 1))
                if axis == 1:
                    lines.append(
                        f"        {name} = jnp.repeat({source}[:, None, :], {repeats!r}, axis=1)"
                    )
                else:
                    lines.append(
                        f"        {name} = jnp.repeat({source}, {repeats!r}, axis={axis!r})"
                    )
            elif definition == "layer.prelu":
                initializer = params.get("alpha_initializer", "zeros")
                initializer_expression = self._initializer_expression(initializer, "zeros")
                shared_axes = params.get("shared_axes")
                lines.append(f"        {name}_alpha_shape = list({source}.shape[1:])")
                if isinstance(shared_axes, list):
                    for shared_axis in shared_axes:
                        lines.append(f"        {name}_alpha_shape[{int(shared_axis) - 1}] = 1")
                lines.append(
                    f"        {name}_alpha = self.param({name + '_alpha'!r}, {initializer_expression}, tuple({name}_alpha_shape))"
                )
                lines.append(
                    f"        {name} = jnp.where({source} >= 0, {source}, {source} * {name}_alpha.reshape((1,) + tuple({name}_alpha_shape)))"
                )
            elif definition == "layer.rms_norm":
                axis = int(params.get("axis", -1))
                lines.append(
                    f"        {name}_scale = self.param({name + '_scale'!r}, nn.initializers.ones, ({source}.shape[{axis}],))"
                )
                lines.append(
                    f"        {name} = {source} / jnp.sqrt(jnp.mean({source} * {source}, axis={axis!r}, keepdims=True) + {float(params.get('epsilon', 1e-6))!r})"
                )
                lines.append(f"        {name}_scale_shape = [1] * {source}.ndim")
                lines.append(f"        {name}_scale_shape[{axis}] = -1")
                lines.append(f"        {name} = {name} * {name}_scale.reshape(tuple({name}_scale_shape))")
            elif definition == "layer.unit_norm":
                axis = int(params.get("axis", -1))
                lines.append(
                    f"        {name} = {source} / jnp.maximum(jnp.linalg.norm({source}, axis={axis!r}, keepdims=True), jnp.asarray(1e-12, dtype={source}.dtype))"
                )
            elif definition == "layer.group_norm":
                axis = int(params.get("axis", -1))
                groups = int(params.get("groups", 32))
                gamma_expression = self._initializer_expression(
                    params.get("gamma_initializer", "ones"), "ones"
                )
                beta_expression = self._initializer_expression(
                    params.get("beta_initializer", "zeros"), "zeros"
                )
                lines.append(f"        {name}_current = jnp.moveaxis({source}, {axis}, -1)")
                lines.append(
                    f"        {name}_grouped = {name}_current.reshape({name}_current.shape[:-1] + ({groups}, {name}_current.shape[-1] // {groups}))"
                )
                lines.append(
                    f"        {name}_reduction_axes = tuple(index for index in range(1, {name}_grouped.ndim) if index != {name}_grouped.ndim - 2)"
                )
                lines.append(
                    f"        {name}_mean = jnp.mean({name}_grouped, axis={name}_reduction_axes, keepdims=True)"
                )
                lines.append(
                    f"        {name}_variance = jnp.mean(({name}_grouped - {name}_mean) * ({name}_grouped - {name}_mean), axis={name}_reduction_axes, keepdims=True)"
                )
                lines.append(
                    f"        {name}_current = (({name}_grouped - {name}_mean) / jnp.sqrt({name}_variance + {float(params.get('epsilon', 0.001))!r})).reshape({name}_current.shape)"
                )
                lines.append(
                    f"        {name}_gamma = self.param({name + '_gamma'!r}, {gamma_expression}, ({name}_current.shape[-1],))"
                    if bool(params.get("scale", True))
                    else f"        {name}_gamma = None"
                )
                lines.append(
                    f"        {name}_beta = self.param({name + '_beta'!r}, {beta_expression}, ({name}_current.shape[-1],))"
                    if bool(params.get("center", True))
                    else f"        {name}_beta = None"
                )
                lines.append(
                    f"        if {name}_gamma is not None: {name}_current = {name}_current * {name}_gamma.reshape((1,) * ({name}_current.ndim - 1) + ({name}_current.shape[-1],))"
                )
                lines.append(
                    f"        if {name}_beta is not None: {name}_current = {name}_current + {name}_beta.reshape((1,) * ({name}_current.ndim - 1) + ({name}_current.shape[-1],))"
                )
                lines.append(f"        {name} = jnp.moveaxis({name}_current, -1, {axis})")
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
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(f"        {name} = sum([{', '.join(source_vars)}])")
            elif definition == "merge.concatenate":
                lines.append(
                    f"        {name} = jnp.concatenate([{', '.join(source_vars)}], axis={int(params.get('axis', -1))!r})"
                )
            elif manifest.semantic_role == "output_adapter":
                lines.append(f"        {name} = {source}")
            else:
                raise ValueError(f"No native JAX adapter registered for {definition}")
            output_ports = [
                port
                for port in materialized_outputs(manifest, params)
                if condition_matches(port.enabled_when, params)
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
        trace_entries: list[str] = []
        for trace_item in ir.ordered_nodes:
            trace_manifest = catalog.get(
                trace_item.definition_id,
                nodes[trace_item.node_id].definition_version,
            )
            trace_ports = ", ".join(
                f"{port.id!r}: {port_variables[(trace_item.node_id, port.id)]}"
                for port in materialized_outputs(
                    trace_manifest,
                    nodes[trace_item.node_id].parameters,
                )
                if condition_matches(port.enabled_when, nodes[trace_item.node_id].parameters)
                if (trace_item.node_id, port.id) in port_variables
            )
            trace_entries.append(
                f"{trace_item.node_id!r}: {{'outputs': {{{trace_ports}}}}}"
            )
        trace = "{" + ", ".join(trace_entries) + "}"
        lines.append(f"        if return_intermediates: return {result}, {trace}")
        lines.append(f"        return {result}")
        jit_enabled = bool(
            graph.model_configuration.native.jax.compiler.parameters.get("enabled", False)
        )
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
