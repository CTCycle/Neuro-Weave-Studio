# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from typing import Any, cast

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import (
    CompilationCatalog,
    CompilationOption,
    get_compilation_catalog,
)
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from domain.metric_formula import MetricFormulaError, emit_metric_formula, parse_metric_formula
from exporters.base import SourceExporter
from validation.output_resolution import output_selectors, resolved_outputs


###############################################################################
class KerasSourceExporter(SourceExporter):

    # -------------------------------------------------------------------------
    @staticmethod
    def _option(value: object) -> object:
        return None if value in (None, "", "none") else value

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
        configuration = graph.model_configuration.keras
        if "initializer" in key:
            return cls._option(configuration.default_initializer)
        if "regularizer" in key:
            return cls._option(configuration.default_regularizer)
        if "constraint" in key:
            return cls._option(configuration.default_constraint)
        return cls._option(fallback)

    # -------------------------------------------------------------------------
    @classmethod
    def _constructor_kwargs(
        cls,
        parameters: dict[str, Any],
        graph: ModelGraph,
        values: dict[str, Any] | None = None,
    ) -> str:
        """Render explicit layer kwargs while preserving Keras defaults."""

        resolved = dict(values or {})
        for key in ("dtype", "trainable", "autocast"):
            value = parameters.get(key)
            if value is not None:
                resolved[key] = value
        return ", ".join(f"{key}={value!r}" for key, value in resolved.items())

    # -------------------------------------------------------------------------
    @classmethod
    def _recurrent_kwargs(
        cls,
        parameters: dict[str, Any],
        graph: ModelGraph,
        *,
        cell: str,
        name: str,
        return_sequences: bool | None = None,
        return_state: bool = False,
    ) -> tuple[str, str]:
        values: dict[str, Any] = {
            "units": parameters.get("units", 64),
            "activation": parameters.get("activation", "tanh"),
            "use_bias": parameters.get("use_bias", True),
            "kernel_initializer": cls._layer_option(
                parameters, "kernel_initializer", graph, "glorot_uniform"
            ),
            "recurrent_initializer": cls._layer_option(
                parameters, "recurrent_initializer", graph, "orthogonal"
            ),
            "bias_initializer": cls._layer_option(
                parameters, "bias_initializer", graph, "zeros"
            ),
            "kernel_regularizer": cls._layer_option(
                parameters, "kernel_regularizer", graph, "none"
            ),
            "recurrent_regularizer": cls._layer_option(
                parameters, "recurrent_regularizer", graph, "none"
            ),
            "bias_regularizer": cls._layer_option(
                parameters, "bias_regularizer", graph, "none"
            ),
            "activity_regularizer": cls._layer_option(
                parameters, "activity_regularizer", graph, "none"
            ),
            "kernel_constraint": cls._layer_option(
                parameters, "kernel_constraint", graph, "none"
            ),
            "recurrent_constraint": cls._layer_option(
                parameters, "recurrent_constraint", graph, "none"
            ),
            "bias_constraint": cls._layer_option(
                parameters, "bias_constraint", graph, "none"
            ),
            "dropout": parameters.get("dropout", 0.0),
            "recurrent_dropout": parameters.get("recurrent_dropout", 0.0),
            "return_sequences": (
                parameters.get("return_sequences", False)
                if return_sequences is None
                else return_sequences
            ),
            "return_state": return_state,
            "go_backwards": parameters.get("go_backwards", False),
            "stateful": parameters.get("stateful", False),
            "unroll": parameters.get("unroll", False),
            "seed": parameters.get("seed"),
            "name": name,
        }
        if cell != "simple_rnn":
            values["recurrent_activation"] = parameters.get("recurrent_activation", "sigmoid")
        if cell == "gru":
            values["reset_after"] = parameters.get("reset_after", True)
            values["use_cudnn"] = parameters.get("use_cudnn", "auto")
            layer_name = "GRU"
        elif cell == "lstm":
            values["unit_forget_bias"] = parameters.get("unit_forget_bias", True)
            values["use_cudnn"] = parameters.get("use_cudnn", "auto")
            layer_name = "LSTM"
        else:
            layer_name = "SimpleRNN"
        return layer_name, cls._constructor_kwargs(parameters, graph, values)

    # -------------------------------------------------------------------------
    @classmethod
    def _compile_mapping_source(
        cls,
        graph: ModelGraph,
        configuration: dict[str, Any],
        default: Any,
        category: str,
        compilation_catalog: CompilationCatalog,
    ) -> Any:
        if not configuration:
            return "None"
        outputs = [
            output
            for output in resolved_outputs(graph, get_node_catalog())
            if output.node is not None
        ]
        values: list[str] = []
        for output in outputs:
            node = output.node
            assert node is not None
            value = configuration.get("__default__", default)
            for selector in output_selectors(output):
                if selector and selector in configuration:
                    value = configuration[selector]
                    break
            if category == "losses":
                if not isinstance(value, str):
                    values.append("None")
                else:
                    option = cls._catalog_option(compilation_catalog, category, value)
                    symbol = cls._keras_value_source(option)
                    values.append(symbol)
            else:
                metric_expressions: list[str] = []
                metric_values = cast(list[Any], value) if isinstance(value, list) else []
                for metric in metric_values:
                    if not isinstance(metric, str):
                        raise ValueError(
                            f"Keras metric identifiers must be strings, got {metric!r}"
                        )
                    option = cls._catalog_option(compilation_catalog, category, metric)
                    metric_expressions.append(cls._keras_value_source(option, metric=True))
                values.append("[" + ", ".join(metric_expressions) + "]")
        if len(values) == 1:
            return values[0]
        return "[" + ", ".join(values) + "]"

    # -------------------------------------------------------------------------
    @staticmethod
    def _catalog_option(catalog: CompilationCatalog, category: str, identifier: str) -> Any:
        options = getattr(catalog, category)
        option = next((item for item in options if item.id == identifier), None)
        if option is None:
            raise ValueError(f"No Keras catalog option registered for {category}.{identifier}")
        implementation = option.implementations.get("keras")
        if implementation is None or not implementation.supported:
            raise ValueError(f"No Keras implementation registered for {category}.{identifier}")
        return option

    # -------------------------------------------------------------------------
    @staticmethod
    def _custom_function_name(option: CompilationOption) -> str:
        return "nw_custom_" + "".join(char if char.isalnum() else "_" for char in option.id)

    # -------------------------------------------------------------------------
    @classmethod
    def _keras_value_source(cls, option: CompilationOption, *, metric: bool = False) -> str:
        implementation = option.implementations.get("keras")
        if option.origin == "custom":
            return cls._custom_function_name(option)
        if implementation is None or not implementation.symbol:
            raise ValueError(f"No Keras symbol registered for {option.id}")
        namespace = "metrics" if metric else "losses"
        return f"keras.{namespace}.{implementation.symbol}()"

    # -------------------------------------------------------------------------
    @classmethod
    def _custom_formula_functions(
        cls, graph: ModelGraph, compilation_catalog: CompilationCatalog
    ) -> list[str]:
        identifiers: set[str] = set()
        for values in (graph.model_configuration.keras.losses, graph.model_configuration.keras.metrics):
            for value in values.values():
                if isinstance(value, str):
                    identifiers.add(value)
                elif isinstance(value, list):
                    identifiers.update(item for item in value if isinstance(item, str))
        options = [
            option
            for option in (*compilation_catalog.losses, *compilation_catalog.metrics)
            if option.id in identifiers and option.origin == "custom"
        ]
        lines: list[str] = []
        for option in options:
            try:
                expression = emit_metric_formula(parse_metric_formula(option.formula or ""), "keras")
            except MetricFormulaError as exc:
                raise ValueError(f"Invalid custom formula for {option.id}: {exc}") from exc
            lines.extend([
                "",
                "",
                f"@keras.saving.register_keras_serializable(package='NeuroWeave', name='{cls._custom_function_name(option)}')",
                f"def {cls._custom_function_name(option)}(y_true, y_pred):",
                f"    return {expression}",
            ])
        return lines

    # -------------------------------------------------------------------------
    @classmethod
    def _optimizer_name(cls, name: str, compilation_catalog: CompilationCatalog) -> str:
        return (
            cls._catalog_option(compilation_catalog, "optimizers", name)
            .implementations["keras"]
            .symbol
            or ""
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _optimizer_parameters(
        cls, graph: ModelGraph, compilation_catalog: CompilationCatalog
    ) -> str:
        optimizer = cls._catalog_option(
            compilation_catalog,
            "optimizers",
            graph.model_configuration.keras.optimizer,
        )
        accepted = {parameter.id for parameter in optimizer.parameters}
        parameters = {
            key: value
            for key, value in graph.model_configuration.keras.optimizer_parameters.items()
            if key in accepted
        }
        return ", ".join(f"{key}={value!r}" for key, value in parameters.items())

    # -------------------------------------------------------------------------
    def generate(
        self,
        graph: ModelGraph,
        ir: CanonicalModelIR,
        catalog: NodeCatalog | None = None,
        *,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> str:
        catalog = catalog or get_node_catalog()
        compilation_catalog = compilation_catalog or get_compilation_catalog()
        nodes = {node.id: node for node in graph.nodes}
        backend = (
            "torch"
            if ir.execution_backend is ExecutionBackend.PYTORCH
            else ir.execution_backend.value
        )
        lines = [
            "from __future__ import annotations",
            "",
            "import os",
            f"os.environ['KERAS_BACKEND'] = {backend!r}",
            "",
            "import keras",
            "from keras import layers",
        ]
        if any(node.definition_id in {"input.constant", "input.parameter"} for node in graph.nodes):
            lines.extend(
                [
                    "",
                    "",
                    "@keras.saving.register_keras_serializable(package='NeuroWeave')",
                    "class GeneratedConstant(layers.Layer):",
                    "    def __init__(self, shape, value, dtype='float32', **kwargs):",
                    "        super().__init__(dtype=dtype, trainable=False, **kwargs)",
                    "        self.source_shape = tuple(shape)",
                    "        self.source_value = value",
                    "    def call(self, anchor):",
                    "        batch = keras.ops.shape(anchor)[0]",
                    "        value = keras.ops.full(self.source_shape, self.source_value, dtype=self.compute_dtype)",
                    "        return keras.ops.repeat(keras.ops.expand_dims(value, 0), batch, axis=0)",
                    "    def get_config(self):",
                    "        return {**super().get_config(), 'shape': list(self.source_shape), 'value': self.source_value}",
                    "",
                    "",
                    "@keras.saving.register_keras_serializable(package='NeuroWeave')",
                    "class GeneratedParameter(layers.Layer):",
                    "    def __init__(self, shape, initializer='zeros', dtype='float32', trainable=True, **kwargs):",
                    "        super().__init__(dtype=dtype, trainable=trainable, **kwargs)",
                    "        self.source_shape = tuple(shape)",
                    "        self.initializer = keras.initializers.get(initializer)",
                    "        self.parameter_trainable = bool(trainable)",
                    "    def build(self, input_shape):",
                    "        del input_shape",
                    "        self.value = self.add_weight(name='value', shape=self.source_shape, initializer=self.initializer, trainable=self.parameter_trainable, dtype=self.compute_dtype)",
                    "    def call(self, anchor):",
                    "        batch = keras.ops.shape(anchor)[0]",
                    "        return keras.ops.repeat(keras.ops.expand_dims(self.value, 0), batch, axis=0)",
                    "    def get_config(self):",
                    "        return {**super().get_config(), 'shape': list(self.source_shape), 'initializer': keras.initializers.serialize(self.initializer), 'trainable': self.parameter_trainable}",
                ]
            )
        if any(node.definition_id == "transformer.positional_encoding" for node in graph.nodes):
            lines.extend(
                [
                    "",
                    "",
                    "@keras.saving.register_keras_serializable(package='NeuroWeave')",
                    "class SinusoidalPositionalEncoding(layers.Layer):",
                    "    def __init__(self, sequence_length=None, feature_dim=None, **kwargs):",
                    "        super().__init__(**kwargs)",
                    "        self.sequence_length = sequence_length",
                    "        self.feature_dim = feature_dim",
                    "    def call(self, inputs):",
                    "        input_shape = getattr(inputs, 'shape', None)",
                    "        static_length = input_shape[1] if input_shape is not None and len(input_shape) > 1 else None",
                    "        static_depth = input_shape[2] if input_shape is not None and len(input_shape) > 2 else None",
                    "        length = self.sequence_length or static_length or keras.ops.shape(inputs)[1]",
                    "        depth = self.feature_dim or static_depth or keras.ops.shape(inputs)[2]",
                    "        positions = keras.ops.cast(keras.ops.arange(0, length, 1)[:, None], inputs.dtype)",
                    "        dimensions = keras.ops.cast(keras.ops.arange(0, depth, 1)[None, :], inputs.dtype)",
                    "        depth_value = keras.ops.cast(depth, inputs.dtype)",
                    "        rates = keras.ops.power(10000.0, -(2.0 * keras.ops.floor(dimensions / 2.0)) / depth_value)",
                    "        angles = positions * rates",
                    "        even = keras.ops.cast(keras.ops.mod(keras.ops.arange(0, depth, 1), 2) == 0, inputs.dtype)[None, :]",
                    "        encoding = keras.ops.sin(angles) * even + keras.ops.cos(angles) * (1.0 - even)",
                    "        return inputs + encoding[None, :, :]",
                    "",
                    "    def get_config(self):",
                    "        return {**super().get_config(), 'sequence_length': self.sequence_length, 'feature_dim': self.feature_dim}",
                ]
            )
        if any(
            node.definition_id
            in {
                "transformer.encoder_block",
                "transformer.stack",
                "transformer.feed_forward",
                "transformer.decoder_block",
            }
            for node in graph.nodes
        ):
            lines.extend(
                [
                    "",
                    "",
                    "@keras.saving.register_keras_serializable(package='NeuroWeave')",
                    "class GeluActivation(layers.Layer):",
                    "    def call(self, inputs):",
                    "        half = keras.ops.cast(0.5, inputs.dtype)",
                    "        one = keras.ops.cast(1.0, inputs.dtype)",
                    "        root_two = keras.ops.cast(2.0, inputs.dtype)",
                    "        return half * inputs * (one + keras.ops.erf(inputs / keras.ops.sqrt(root_two)))",
                    "",
                    "    def get_config(self):",
                    "        return super().get_config()",
                ]
            )
        lines.extend(self._custom_formula_functions(graph, compilation_catalog))
        lines.extend(["", "", "def build_model() -> keras.Model:"])
        if graph.model_configuration.keras.random_seed is not None:
            lines.append(
                f"    keras.utils.set_random_seed({graph.model_configuration.keras.random_seed!r})"
            )
        policy = "mixed_float16" if graph.model_configuration.keras.mixed_precision else "float32"
        lines.append(f"    keras.mixed_precision.set_global_policy({policy!r})")

        variables: dict[str, str] = {}
        port_variables: dict[tuple[str, str], str] = {}
        for input_node_id in ir.input_node_ids:
            input_node = nodes[input_node_id]
            input_variable = next(item.variable_name for item in ir.ordered_nodes if item.node_id == input_node_id)
            shape = list(input_node.parameters.get("shape", [None, 1]))
            feature_shape = shape[1:] if shape and shape[0] is None else shape
            lines.append(
                f"    {input_variable} = keras.Input(shape={feature_shape!r}, name={input_node.id!r}, dtype={input_node.parameters.get('dtype', 'float32')!r})"
            )
            variables[input_node_id] = input_variable
        for item in ir.ordered_nodes:
            node = nodes[item.node_id]
            parameters = node.parameters
            variable = item.variable_name
            variables[item.node_id] = variable
            inputs = [
                port_variables[(binding.source_node_id, binding.source_port_id)]
                for binding in item.input_bindings
            ]
            source = inputs[0] if inputs else ""
            definition = item.definition_id
            manifest = catalog.get(definition, node.definition_version)
            if definition in {"input.constant", "input.parameter"}:
                if not ir.input_node_ids:
                    raise ValueError(
                        f"Generated source node {node.id} requires at least one runtime model input for batch materialization"
                    )
                anchor = variables[ir.input_node_ids[0]]
                source_shape = parameters.get("shape", [1])
                if not isinstance(source_shape, list):
                    raise ValueError(f"Generated source node {node.id} requires a list shape")
                if definition == "input.constant":
                    lines.append(
                        f"    {variable} = GeneratedConstant(shape={source_shape!r}, value={parameters.get('value', 0)!r}, dtype={parameters.get('dtype', 'float32')!r}, name={node.id!r})({anchor})"
                    )
                else:
                    lines.append(
                        f"    {variable} = GeneratedParameter(shape={source_shape!r}, initializer={parameters.get('initializer', 'zeros')!r}, dtype={parameters.get('dtype', 'float32')!r}, trainable={bool(parameters.get('trainable', True))!r}, name={node.id!r})({anchor})"
                    )
            elif manifest.operation is not None:
                lines.append(
                    f"    {variable} = {emit_operation_source(manifest, 'keras', inputs, parameters)}"
                )
            elif manifest.semantic_role == "input":
                if item.node_id not in variables:
                    raise ValueError(f"Runtime input {node.id} was not materialized")
            elif definition == "layer.dense":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "units": parameters.get("units", 32),
                        "activation": parameters.get("activation", "linear"),
                        "use_bias": parameters.get("use_bias", True),
                        "kernel_initializer": self._layer_option(
                            parameters, "kernel_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "kernel_regularizer": self._layer_option(
                            parameters, "kernel_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "kernel_constraint": self._layer_option(
                            parameters, "kernel_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "lora_rank": parameters.get("lora_rank"),
                        "lora_alpha": parameters.get("lora_alpha"),
                        "quantization_config": parameters.get("quantization_config"),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.Dense({kwargs})({source})"
                )
            elif definition == "layer.activation":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {"activation": parameters.get("activation", "relu"), "name": node.id},
                )
                lines.append(
                    f"    {variable} = layers.Activation({kwargs})({source})"
                )
            elif definition == "layer.prelu":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "alpha_initializer": self._layer_option(
                            parameters, "alpha_initializer", graph, "zeros"
                        ),
                        "alpha_regularizer": self._layer_option(
                            parameters, "alpha_regularizer", graph, "none"
                        ),
                        "alpha_constraint": self._layer_option(
                            parameters, "alpha_constraint", graph, "none"
                        ),
                        "shared_axes": parameters.get("shared_axes"),
                        "name": node.id,
                    },
                )
                lines.append(f"    {variable} = layers.PReLU({kwargs})({source})")
            elif definition == "layer.dropout":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "rate": parameters.get("rate", 0.5),
                        "noise_shape": parameters.get("noise_shape"),
                        "seed": parameters.get("seed"),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.Dropout({kwargs})({source})"
                )
            elif definition == "layer.flatten":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {"data_format": parameters.get("data_format", "channels_last"), "name": node.id},
                )
                lines.append(
                    f"    {variable} = layers.Flatten({kwargs})({source})"
                )
            elif definition == "layer.batch_norm":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "axis": parameters.get("axis", -1),
                        "momentum": parameters.get("momentum", 0.99),
                        "epsilon": parameters.get("epsilon", 0.001),
                        "center": parameters.get("center", True),
                        "scale": parameters.get("scale", True),
                        "beta_initializer": self._layer_option(
                            parameters, "beta_initializer", graph, "zeros"
                        ),
                        "gamma_initializer": self._layer_option(
                            parameters, "gamma_initializer", graph, "ones"
                        ),
                        "moving_mean_initializer": self._layer_option(
                            parameters, "moving_mean_initializer", graph, "zeros"
                        ),
                        "moving_variance_initializer": self._layer_option(
                            parameters, "moving_variance_initializer", graph, "ones"
                        ),
                        "beta_regularizer": self._layer_option(
                            parameters, "beta_regularizer", graph, "none"
                        ),
                        "gamma_regularizer": self._layer_option(
                            parameters, "gamma_regularizer", graph, "none"
                        ),
                        "beta_constraint": self._layer_option(
                            parameters, "beta_constraint", graph, "none"
                        ),
                        "gamma_constraint": self._layer_option(
                            parameters, "gamma_constraint", graph, "none"
                        ),
                        "renorm": parameters.get("renorm", False),
                        "renorm_clipping": parameters.get("renorm_clipping"),
                        "renorm_momentum": parameters.get("renorm_momentum", 0.99),
                        "synchronized": parameters.get("synchronized", False),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.BatchNormalization({kwargs})({source})"
                )
            elif definition == "layer.layer_norm":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "axis": parameters.get("axis", [-1]),
                        "epsilon": parameters.get("epsilon", 0.001),
                        "center": parameters.get("center", True),
                        "scale": parameters.get("scale", True),
                        "beta_initializer": self._layer_option(
                            parameters, "beta_initializer", graph, "zeros"
                        ),
                        "gamma_initializer": self._layer_option(
                            parameters, "gamma_initializer", graph, "ones"
                        ),
                        "beta_regularizer": self._layer_option(
                            parameters, "beta_regularizer", graph, "none"
                        ),
                        "gamma_regularizer": self._layer_option(
                            parameters, "gamma_regularizer", graph, "none"
                        ),
                        "beta_constraint": self._layer_option(
                            parameters, "beta_constraint", graph, "none"
                        ),
                        "gamma_constraint": self._layer_option(
                            parameters, "gamma_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.LayerNormalization({kwargs})({source})"
                )
            elif definition == "layer.rms_norm":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "axis": parameters.get("axis", -1),
                        "epsilon": parameters.get("epsilon", 1e-6),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.RMSNormalization({kwargs})({source})"
                )
            elif definition == "layer.unit_norm":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {"axis": parameters.get("axis", -1), "name": node.id},
                )
                lines.append(
                    f"    {variable} = layers.UnitNormalization({kwargs})({source})"
                )
            elif definition == "layer.group_norm":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "groups": parameters.get("groups", 32),
                        "axis": parameters.get("axis", -1),
                        "epsilon": parameters.get("epsilon", 0.001),
                        "center": parameters.get("center", True),
                        "scale": parameters.get("scale", True),
                        "beta_initializer": self._layer_option(
                            parameters, "beta_initializer", graph, "zeros"
                        ),
                        "gamma_initializer": self._layer_option(
                            parameters, "gamma_initializer", graph, "ones"
                        ),
                        "beta_regularizer": self._layer_option(
                            parameters, "beta_regularizer", graph, "none"
                        ),
                        "gamma_regularizer": self._layer_option(
                            parameters, "gamma_regularizer", graph, "none"
                        ),
                        "beta_constraint": self._layer_option(
                            parameters, "beta_constraint", graph, "none"
                        ),
                        "gamma_constraint": self._layer_option(
                            parameters, "gamma_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(f"    {variable} = layers.GroupNormalization({kwargs})({source})")
            elif definition == "layer.reshape":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {"name": node.id},
                )
                lines.append(
                    f"    {variable} = layers.Reshape({tuple(parameters.get('target_shape', [-1]))!r}, {kwargs})({source})"
                )
            elif definition == "layer.permute":
                kwargs = self._constructor_kwargs(parameters, graph, {"name": node.id})
                lines.append(
                    f"    {variable} = layers.Permute({tuple(parameters.get('axes', [2, 1]))!r}, {kwargs})({source})"
                )
            elif definition == "layer.repeat":
                repeats = int(parameters.get("repeats", 2))
                axis = int(parameters.get("axis", 1))
                kwargs = self._constructor_kwargs(parameters, graph, {"name": node.id})
                if axis == 1:
                    lines.append(
                        f"    {variable} = layers.RepeatVector({repeats!r}, {kwargs})({source})"
                    )
                else:
                    lines.append(
                        f"    {variable} = layers.Lambda(lambda value: keras.ops.repeat(value, {repeats!r}, axis={axis!r}), {kwargs})({source})"
                    )
            elif definition == "layer.embedding":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "input_dim": parameters.get("vocabulary_size", 10000),
                        "output_dim": parameters.get("embedding_dim", 128),
                        "embeddings_initializer": self._layer_option(
                            parameters, "embeddings_initializer", graph, "uniform"
                        ),
                        "embeddings_regularizer": self._layer_option(
                            parameters, "embeddings_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "embeddings_constraint": self._layer_option(
                            parameters, "embeddings_constraint", graph, "none"
                        ),
                        "mask_zero": parameters.get("mask_zero", False),
                        "lora_rank": parameters.get("lora_rank"),
                        "lora_alpha": parameters.get("lora_alpha"),
                        "quantization_config": parameters.get("quantization_config"),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.Embedding({kwargs})({source})"
                )
            elif definition == "conv.conv1d":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "filters": parameters.get("filters", 32),
                        "kernel_size": parameters.get("kernel_size", [3]),
                        "strides": parameters.get("strides", [1]),
                        "padding": parameters.get("padding", "valid"),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "dilation_rate": parameters.get("dilation_rate", [1]),
                        "groups": parameters.get("groups", 1),
                        "activation": parameters.get("activation", "linear"),
                        "use_bias": parameters.get("use_bias", True),
                        "kernel_initializer": self._layer_option(
                            parameters, "kernel_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "kernel_regularizer": self._layer_option(
                            parameters, "kernel_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "kernel_constraint": self._layer_option(
                            parameters, "kernel_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.Conv1D({kwargs})({source})"
                )
            elif definition == "conv.conv2d":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "filters": parameters.get("filters", 32),
                        "kernel_size": parameters.get("kernel_size", [3, 3]),
                        "strides": parameters.get("strides", [1, 1]),
                        "padding": parameters.get("padding", "valid"),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "dilation_rate": parameters.get("dilation_rate", [1, 1]),
                        "groups": parameters.get("groups", 1),
                        "activation": parameters.get("activation", "linear"),
                        "use_bias": parameters.get("use_bias", True),
                        "kernel_initializer": self._layer_option(
                            parameters, "kernel_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "kernel_regularizer": self._layer_option(
                            parameters, "kernel_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "kernel_constraint": self._layer_option(
                            parameters, "kernel_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.Conv2D({kwargs})({source})"
                )
            elif definition == "conv.max_pool2d":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "pool_size": parameters.get("pool_size", [2, 2]),
                        "strides": parameters.get("strides"),
                        "padding": parameters.get("padding", "valid"),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.MaxPooling2D({kwargs})({source})"
                )
            elif definition == "conv.conv3d":
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "filters": parameters.get("filters", 32),
                        "kernel_size": parameters.get("kernel_size", [3, 3, 3]),
                        "strides": parameters.get("strides", [1, 1, 1]),
                        "padding": parameters.get("padding", "valid"),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "dilation_rate": parameters.get("dilation_rate", [1, 1, 1]),
                        "groups": parameters.get("groups", 1),
                        "activation": parameters.get("activation", "linear"),
                        "use_bias": parameters.get("use_bias", True),
                        "kernel_initializer": self._layer_option(
                            parameters, "kernel_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "kernel_regularizer": self._layer_option(
                            parameters, "kernel_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "kernel_constraint": self._layer_option(
                            parameters, "kernel_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(f"    {variable} = layers.Conv3D({kwargs})({source})")
            elif definition in {
                "conv.max_pool1d",
                "conv.max_pool3d",
                "conv.average_pool1d",
                "conv.average_pool2d",
                "conv.average_pool3d",
            }:
                dimensions = int(definition[-2])
                pooling = "MaxPooling" if definition.startswith("conv.max") else "AveragePooling"
                layer_name = f"{pooling}{dimensions}D"
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "pool_size": parameters.get("pool_size", [2] * dimensions),
                        "strides": parameters.get("strides"),
                        "padding": parameters.get("padding", "valid"),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "name": node.id,
                    },
                )
                lines.append(f"    {variable} = layers.{layer_name}({kwargs})({source})")
            elif definition in {
                "conv.global_pool1d",
                "conv.global_pool2d",
                "conv.global_pool3d",
            }:
                dimensions = int(definition[-2])
                mode = parameters.get("mode", "average")
                pooling = "GlobalAveragePooling" if mode == "average" else "GlobalMaxPooling"
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "data_format": parameters.get("data_format", "channels_last"),
                        "keepdims": parameters.get("keepdims", False),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.{pooling}{dimensions}D({kwargs})({source})"
                )
            elif definition == "conv.transposed":
                dimensions = int(parameters.get("dimensions", 2))
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "filters": parameters.get("filters", 32),
                        "kernel_size": parameters.get("kernel_size", [3] * dimensions),
                        "strides": parameters.get("strides", [1] * dimensions),
                        "padding": parameters.get("padding", "valid"),
                        "output_padding": parameters.get("output_padding"),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "dilation_rate": parameters.get("dilation_rate", [1] * dimensions),
                        "activation": parameters.get("activation", "linear"),
                        "use_bias": parameters.get("use_bias", True),
                        "kernel_initializer": self._layer_option(
                            parameters, "kernel_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "kernel_regularizer": self._layer_option(
                            parameters, "kernel_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "kernel_constraint": self._layer_option(
                            parameters, "kernel_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(f"    {variable} = layers.Conv{dimensions}DTranspose({kwargs})({source})")
            elif definition == "conv.depthwise":
                dimensions = int(parameters.get("dimensions", 2))
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "kernel_size": parameters.get("kernel_size", [3] * dimensions),
                        "strides": parameters.get("strides", [1] * dimensions),
                        "padding": parameters.get("padding", "valid"),
                        "depth_multiplier": parameters.get("depth_multiplier", 1),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "dilation_rate": parameters.get("dilation_rate", [1] * dimensions),
                        "activation": parameters.get("activation", "linear"),
                        "use_bias": parameters.get("use_bias", True),
                        "depthwise_initializer": self._layer_option(
                            parameters, "depthwise_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "depthwise_regularizer": self._layer_option(
                            parameters, "depthwise_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "depthwise_constraint": self._layer_option(
                            parameters, "depthwise_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(f"    {variable} = layers.DepthwiseConv{dimensions}D({kwargs})({source})")
            elif definition == "conv.separable":
                dimensions = int(parameters.get("dimensions", 2))
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "filters": parameters.get("filters", 32),
                        "kernel_size": parameters.get("kernel_size", [3] * dimensions),
                        "strides": parameters.get("strides", [1] * dimensions),
                        "padding": parameters.get("padding", "valid"),
                        "data_format": parameters.get("data_format", "channels_last"),
                        "dilation_rate": parameters.get("dilation_rate", [1] * dimensions),
                        "depth_multiplier": parameters.get("depth_multiplier", 1),
                        "activation": parameters.get("activation", "linear"),
                        "use_bias": parameters.get("use_bias", True),
                        "depthwise_initializer": self._layer_option(
                            parameters, "depthwise_initializer", graph, "glorot_uniform"
                        ),
                        "pointwise_initializer": self._layer_option(
                            parameters, "pointwise_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "depthwise_regularizer": self._layer_option(
                            parameters, "depthwise_regularizer", graph, "none"
                        ),
                        "pointwise_regularizer": self._layer_option(
                            parameters, "pointwise_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "depthwise_constraint": self._layer_option(
                            parameters, "depthwise_constraint", graph, "none"
                        ),
                        "pointwise_constraint": self._layer_option(
                            parameters, "pointwise_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "name": node.id,
                    },
                )
                lines.append(f"    {variable} = layers.SeparableConv{dimensions}D({kwargs})({source})")
            elif definition == "conv.upsampling":
                dimensions = int(parameters.get("dimensions", 2))
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "size": parameters.get("size", [2] * dimensions),
                        "name": node.id,
                    },
                )
                if dimensions > 1:
                    kwargs = f"{kwargs}, data_format={parameters.get('data_format', 'channels_last')!r}"
                if dimensions == 2:
                    kwargs = f"{kwargs}, interpolation={parameters.get('interpolation', 'nearest')!r}"
                lines.append(f"    {variable} = layers.UpSampling{dimensions}D({kwargs})({source})")
            elif definition in {"conv.padding", "conv.cropping"}:
                dimensions = int(parameters.get("dimensions", 2))
                prefix = "ZeroPadding" if definition == "conv.padding" else "Cropping"
                key = "padding" if definition == "conv.padding" else "cropping"
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {key: parameters.get(key), "name": node.id},
                )
                if dimensions > 1:
                    kwargs = f"{kwargs}, data_format={parameters.get('data_format', 'channels_last')!r}"
                lines.append(f"    {variable} = layers.{prefix}{dimensions}D({kwargs})({source})")
            elif definition == "recurrent.simple_rnn":
                layer_name, kwargs = self._recurrent_kwargs(
                    parameters,
                    graph,
                    cell="simple_rnn",
                    name=node.id,
                    return_state=bool(parameters.get("return_state", False)),
                )
                lines.append(
                    f"    {variable} = layers.{layer_name}({kwargs})({source})"
                )
            elif definition == "recurrent.lstm":
                layer_name, kwargs = self._recurrent_kwargs(
                    parameters,
                    graph,
                    cell="lstm",
                    name=node.id,
                    return_state=bool(parameters.get("return_state", False)),
                )
                lines.append(
                    f"    {variable} = layers.{layer_name}({kwargs})({source})"
                )
            elif definition == "recurrent.gru":
                layer_name, kwargs = self._recurrent_kwargs(
                    parameters,
                    graph,
                    cell="gru",
                    name=node.id,
                    return_state=bool(parameters.get("return_state", False)),
                )
                lines.append(
                    f"    {variable} = layers.{layer_name}({kwargs})({source})"
                )
            elif definition == "recurrent.bidirectional":
                cell = str(parameters.get("cell", "lstm"))
                inner_name, inner_kwargs = self._recurrent_kwargs(
                    parameters,
                    graph,
                    cell=cell,
                    name=f"{node.id}_inner",
                    return_state=False,
                )
                wrapper_kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "merge_mode": parameters.get("merge_mode", "concat"),
                        "name": node.id,
                    },
                )
                lines.extend(
                    [
                        f"    {variable}_inner = layers.{inner_name}({inner_kwargs})",
                        f"    {variable} = layers.Bidirectional({variable}_inner, {wrapper_kwargs})({source})",
                    ]
                )
            elif definition == "recurrent.stack":
                cell = str(parameters.get("cell", "lstm"))
                layer_count = int(parameters.get("layers", 2))
                current = source
                for index in range(layer_count):
                    inner_name, inner_kwargs = self._recurrent_kwargs(
                        parameters,
                        graph,
                        cell=cell,
                        name=f"{node.id}_{index + 1}",
                        return_sequences=(
                            True
                            if index < layer_count - 1
                            else bool(parameters.get("return_sequences", False))
                        ),
                        return_state=False,
                    )
                    current_name = f"{variable}_{index + 1}"
                    lines.append(
                        f"    {current_name} = layers.{inner_name}({inner_kwargs})({current})"
                    )
                    current = current_name
                lines.append(f"    {variable} = {current}")
            elif definition in {"attention.multi_head", "attention.self", "attention.cross"}:
                query = inputs[0]
                value = inputs[1] if len(inputs) > 1 else query
                kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "num_heads": parameters.get("num_heads", 8),
                        "key_dim": parameters.get("key_dim", 64),
                        "value_dim": parameters.get("value_dim"),
                        "dropout": parameters.get("dropout", 0.0),
                        "use_bias": parameters.get("use_bias", True),
                        "output_shape": parameters.get("output_shape"),
                        "attention_axes": parameters.get("attention_axes"),
                        "sliding_window": parameters.get("sliding_window"),
                        "flash_attention": parameters.get("flash_attention"),
                        "use_gate": parameters.get("use_gate", False),
                        "kernel_initializer": self._layer_option(
                            parameters, "kernel_initializer", graph, "glorot_uniform"
                        ),
                        "bias_initializer": self._layer_option(
                            parameters, "bias_initializer", graph, "zeros"
                        ),
                        "kernel_regularizer": self._layer_option(
                            parameters, "kernel_regularizer", graph, "none"
                        ),
                        "bias_regularizer": self._layer_option(
                            parameters, "bias_regularizer", graph, "none"
                        ),
                        "activity_regularizer": self._layer_option(
                            parameters, "activity_regularizer", graph, "none"
                        ),
                        "kernel_constraint": self._layer_option(
                            parameters, "kernel_constraint", graph, "none"
                        ),
                        "bias_constraint": self._layer_option(
                            parameters, "bias_constraint", graph, "none"
                        ),
                        "seed": parameters.get("seed"),
                        "name": node.id,
                    },
                )
                lines.append(
                    f"    {variable} = layers.MultiHeadAttention({kwargs})({query}, {value})"
                )
            elif definition in {"merge.add", "merge.residual", "merge.skip"}:
                lines.append(
                    f"    {variable} = layers.Add(name={node.id!r})([{', '.join(inputs)}])"
                )
            elif definition == "merge.concatenate":
                lines.append(
                    f"    {variable} = layers.Concatenate(axis={parameters.get('axis', -1)!r}, name={node.id!r})([{', '.join(inputs)}])"
                )
            elif definition == "merge.split":
                lines.append(
                    f"    {variable} = list(keras.ops.split({source}, {int(parameters.get('sections', 2))!r}, axis={int(parameters.get('axis', -1))!r}))"
                )
            elif definition == "transformer.positional_encoding":
                lines.append(
                    f"    {variable} = SinusoidalPositionalEncoding(sequence_length={source}.shape[1], feature_dim={source}.shape[2], name={node.id!r})({source})"
                )
            elif definition in {"transformer.encoder_block", "transformer.stack"}:
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
                        attention_norm_kwargs = self._constructor_kwargs(
                            parameters,
                            graph,
                            {"name": attention_norm_name},
                        )
                        attention_kwargs = self._constructor_kwargs(
                            parameters,
                            graph,
                            {
                                "num_heads": parameters.get("num_heads", 8),
                                "key_dim": parameters.get("key_dim", 64),
                                "dropout": parameters.get("dropout", 0.0),
                                "name": attention_name,
                            },
                        )
                        ffn_norm_kwargs = self._constructor_kwargs(
                            parameters,
                            graph,
                            {"name": ffn_norm_name},
                        )
                        lines.extend(
                            [
                                f"    {attention_input} = layers.LayerNormalization({attention_norm_kwargs})({current})",
                                f"    {attention} = layers.MultiHeadAttention({attention_kwargs})({attention_input}, {attention_input})",
                                f"    {attention_residual} = layers.Add(name={residual_name!r})([{current}, {attention}])",
                                f"    {ffn_input} = layers.LayerNormalization({ffn_norm_kwargs})({attention_residual})",
                            ]
                        )
                    else:
                        attention = f"{variable}_attention_{index + 1}"
                        attention_residual = f"{variable}_attention_residual_{index + 1}"
                        ffn_input = f"{variable}_ffn_input_{index + 1}"
                        attention_kwargs = self._constructor_kwargs(
                            parameters,
                            graph,
                            {
                                "num_heads": parameters.get("num_heads", 8),
                                "key_dim": parameters.get("key_dim", 64),
                                "dropout": parameters.get("dropout", 0.0),
                                "name": attention_name,
                            },
                        )
                        attention_norm_kwargs = self._constructor_kwargs(
                            parameters,
                            graph,
                            {"name": attention_norm_name},
                        )
                        lines.extend(
                            [
                                f"    {attention} = layers.MultiHeadAttention({attention_kwargs})({current}, {current})",
                                f"    {attention_residual} = layers.LayerNormalization({attention_norm_kwargs})(layers.Add(name={residual_name!r})([{current}, {attention}]))",
                                f"    {ffn_input} = {attention_residual}",
                            ]
                        )
                    feed_forward = f"{variable}_feed_forward_{index + 1}"
                    feed_forward_projected = f"{variable}_feed_forward_projected_{index + 1}"
                    feed_forward_dropout = f"{variable}_feed_forward_dropout_{index + 1}"
                    current = f"{variable}_output_{index + 1}"
                    expand_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {
                            "activation": None if activation == "gelu" else activation,
                            "name": node.id + suffix + "_ffn_expand",
                        },
                    )
                    project_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {"name": node.id + suffix + "_ffn_project"},
                    )
                    dropout_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {
                            "rate": parameters.get("dropout", 0.0),
                            "name": node.id + suffix + "_ffn_dropout",
                        },
                    )
                    lines.extend(
                        [
                            f"    {feed_forward} = layers.Dense({hidden_dim}, {expand_kwargs})({ffn_input})",
                            f"    {feed_forward_projected} = layers.Dense({model_dim}, {project_kwargs})({feed_forward})",
                            f"    {feed_forward_dropout} = layers.Dropout({dropout_kwargs})({feed_forward_projected})",
                            f"    {current} = layers.Add(name={ffn_residual_name!r})([{attention_residual}, {feed_forward_dropout}])",
                        ]
                    )
                    if activation == "gelu":
                        lines.append(
                            f"    {feed_forward} = GeluActivation(name={node.id + suffix + '_ffn_activation'!r})({feed_forward})"
                        )
                    if not pre_norm:
                        output_norm_kwargs = self._constructor_kwargs(
                            parameters,
                            graph,
                            {"name": node.id + suffix + "_output_norm"},
                        )
                        lines.append(
                            f"    {current} = layers.LayerNormalization({output_norm_kwargs})({current})"
                        )
                lines.append(f"    {variable} = {current}")
            elif definition == "transformer.feed_forward":
                source_dimension = f"int({source}.shape[-1])"
                hidden_dimension = f"{source_dimension} * {int(parameters.get('ffn_multiplier', 4))!r}"
                current = f"{variable}_expand"
                expand_kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "activation": (
                            None
                            if parameters.get("activation", "gelu") == "gelu"
                            else parameters.get("activation", "gelu")
                        ),
                        "name": node.id + "_expand",
                    },
                )
                dropout_kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {
                        "rate": parameters.get("dropout", 0.0),
                        "name": node.id + "_dropout",
                    },
                )
                project_kwargs = self._constructor_kwargs(
                    parameters,
                    graph,
                    {"name": node.id + "_project"},
                )
                lines.extend(
                    [
                        f"    {current} = layers.Dense({hidden_dimension}, {expand_kwargs})({source})",
                        f"    {variable}_dropout = layers.Dropout({dropout_kwargs})({current})",
                        f"    {variable} = layers.Dense({source_dimension}, {project_kwargs})({variable}_dropout)",
                    ]
                )
                if parameters.get("activation", "gelu") == "gelu":
                    lines.insert(
                        len(lines) - 2,
                        f"    {current} = GeluActivation(name={node.id + '_activation'!r})({current})",
                    )
            elif definition == "transformer.decoder_block":
                query = inputs[0]
                context = inputs[1] if len(inputs) > 1 else query
                model_dimension = f"int({query}.shape[-1])"
                current = query
                activation = parameters.get("activation", "gelu")
                multiplier = int(parameters.get("ffn_multiplier", 4))
                pre_norm = bool(parameters.get("pre_norm", True))
                for index in range(int(parameters.get("layers", 1))):
                    suffix = f"_{index + 1}"
                    self_attention = f"{variable}_self_attention_{index + 1}"
                    self_residual = f"{variable}_self_residual_{index + 1}"
                    current_self = f"{variable}_self_norm_{index + 1}"
                    cross_attention = f"{variable}_cross_attention_{index + 1}"
                    cross_residual = f"{variable}_cross_residual_{index + 1}"
                    current_cross = f"{variable}_cross_norm_{index + 1}"
                    feed_forward = f"{variable}_ffn_expand_{index + 1}"
                    feed_forward_dropout = f"{variable}_ffn_dropout_{index + 1}"
                    feed_forward_projected = f"{variable}_ffn_project_{index + 1}"
                    ffn_residual = f"{variable}_ffn_residual_{index + 1}"
                    next_current = f"{variable}_output_{index + 1}"
                    self_input = f"{variable}_self_input_{index + 1}"
                    cross_input = f"{variable}_cross_input_{index + 1}"
                    ffn_input = f"{variable}_ffn_input_{index + 1}"
                    attention_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {
                            "num_heads": parameters.get("num_heads", 8),
                            "key_dim": parameters.get("key_dim", 64),
                            "dropout": parameters.get("dropout", 0.0),
                            "name": node.id + suffix + "_self_attention",
                        },
                    )
                    cross_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {
                            "num_heads": parameters.get("num_heads", 8),
                            "key_dim": parameters.get("key_dim", 64),
                            "dropout": parameters.get("dropout", 0.0),
                            "name": node.id + suffix + "_cross_attention",
                        },
                    )
                    norm_self_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {"name": node.id + suffix + "_self_norm"},
                    )
                    norm_cross_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {"name": node.id + suffix + "_cross_norm"},
                    )
                    norm_ffn_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {"name": node.id + suffix + "_ffn_norm"},
                    )
                    expand_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {
                            "activation": None if activation == "gelu" else activation,
                            "name": node.id + suffix + "_ffn_expand",
                        },
                    )
                    dropout_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {
                            "rate": parameters.get("dropout", 0.0),
                            "name": node.id + suffix + "_ffn_dropout",
                        },
                    )
                    project_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {"name": node.id + suffix + "_ffn_project"},
                    )
                    output_norm_kwargs = self._constructor_kwargs(
                        parameters,
                        graph,
                        {"name": node.id + suffix + "_output_norm"},
                    )
                    if pre_norm:
                        lines.extend(
                            [
                                f"    {self_input} = layers.LayerNormalization({norm_self_kwargs})({current})",
                                f"    {self_attention} = layers.MultiHeadAttention({attention_kwargs})({self_input}, {self_input})",
                                f"    {self_residual} = layers.Add(name={node.id + suffix + '_self_residual'!r})([{current}, {self_attention}])",
                                f"    {current_self} = {self_residual}",
                            ]
                        )
                    else:
                        lines.extend(
                            [
                                f"    {self_attention} = layers.MultiHeadAttention({attention_kwargs})({current}, {current})",
                                f"    {self_residual} = layers.Add(name={node.id + suffix + '_self_residual'!r})([{current}, {self_attention}])",
                                f"    {current_self} = layers.LayerNormalization({norm_self_kwargs})({self_residual})",
                            ]
                        )
                    if pre_norm:
                        lines.extend(
                            [
                                f"    {cross_input} = layers.LayerNormalization({norm_cross_kwargs})({current_self})",
                                f"    {cross_attention} = layers.MultiHeadAttention({cross_kwargs})({cross_input}, {context})",
                                f"    {cross_residual} = layers.Add(name={node.id + suffix + '_cross_residual'!r})([{current_self}, {cross_attention}])",
                                f"    {current_cross} = {cross_residual}",
                                f"    {ffn_input} = layers.LayerNormalization({norm_ffn_kwargs})({current_cross})",
                            ]
                        )
                    else:
                        lines.extend(
                            [
                                f"    {cross_attention} = layers.MultiHeadAttention({cross_kwargs})({current_self}, {context})",
                                f"    {cross_residual} = layers.Add(name={node.id + suffix + '_cross_residual'!r})([{current_self}, {cross_attention}])",
                                f"    {current_cross} = layers.LayerNormalization({norm_cross_kwargs})({cross_residual})",
                                f"    {ffn_input} = {current_cross}",
                            ]
                        )
                    lines.extend(
                        [
                            f"    {feed_forward} = layers.Dense({model_dimension} * {multiplier!r}, {expand_kwargs})({ffn_input})",
                            f"    {feed_forward_dropout} = layers.Dropout({dropout_kwargs})({feed_forward})",
                            f"    {feed_forward_projected} = layers.Dense({model_dimension}, {project_kwargs})({feed_forward_dropout})",
                            f"    {ffn_residual} = layers.Add(name={node.id + suffix + '_ffn_residual'!r})([{current_cross}, {feed_forward_projected}])",
                            f"    {next_current} = {ffn_residual}",
                        ]
                    )
                    if activation == "gelu":
                        lines.insert(
                            len(lines) - 4,
                            f"    {feed_forward} = GeluActivation(name={node.id + suffix + '_ffn_activation'!r})({feed_forward})",
                        )
                    if not pre_norm:
                        lines[-1] = f"    {next_current} = layers.LayerNormalization({output_norm_kwargs})({ffn_residual})"
                    current = next_current
                lines.append(f"    {variable} = {current}")
            elif definition == "transformer.residual":
                lines.append(
                    f"    {variable} = layers.Add({self._constructor_kwargs(parameters, graph, {'name': node.id})})([{', '.join(inputs)}])"
                )
            elif definition == "output.detection":
                lines.append(f"    {variable} = ({', '.join(inputs)})")
            elif definition == "transformer.attention_mask" or manifest.semantic_role == "output_adapter":
                lines.append(
                    f"    {variable} = layers.Activation('linear', {self._constructor_kwargs(parameters, graph, {'name': node.id})})({source})"
                )
            else:
                raise ValueError(f"No Keras adapter registered for {definition}")

            output_ports = [
                port
                for port in manifest.outputs
                if self._condition_matches(port.enabled_when, parameters)
            ]
            if not output_ports:
                raise ValueError(f"No enabled Keras output port registered for {definition}")
            if len(output_ports) > 1:
                lines.append(
                    f"    {variable}_outputs = {variable}"
                )
                for index, port in enumerate(output_ports):
                    port_variables[(item.node_id, port.id)] = f"{variable}_outputs[{index}]"
                variables[item.node_id] = f"{variable}_outputs[0]"
            else:
                port_variables[(item.node_id, output_ports[0].id)] = variable
                variables[item.node_id] = variable

        input_variables = ", ".join(variables[node_id] for node_id in ir.input_node_ids)
        output_variables = ", ".join(
            port_variables[(binding.node_id, binding.port_id)] for binding in ir.output_bindings
        )
        lines.append(
            f"    model = keras.Model(inputs=[{input_variables}], outputs=[{output_variables}], name='neuro_weave_model')"
        )
        optimizer_name = self._optimizer_name(
            graph.model_configuration.keras.optimizer, compilation_catalog
        )
        loss = self._compile_mapping_source(
            graph,
            graph.model_configuration.keras.losses,
            None,
            "losses",
            compilation_catalog,
        )
        metrics = self._compile_mapping_source(
            graph,
            graph.model_configuration.keras.metrics,
            [],
            "metrics",
            compilation_catalog,
        )
        optimizer_parameters = self._optimizer_parameters(graph, compilation_catalog)
        lines.extend(
            [
                f"    optimizer = keras.optimizers.{optimizer_name}({optimizer_parameters})",
                f"    model.compile(optimizer=optimizer, loss={loss}, metrics={metrics})",
                "    return model",
                "",
                "",
                "if __name__ == '__main__':",
                "    model = build_model()",
                "    model.summary()",
            ]
        )
        return "\n".join(lines) + "\n"
