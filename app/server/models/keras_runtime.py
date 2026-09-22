# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

from typing import Any

import keras

from common.catalogs.loader import get_node_catalog
from domain.compilation_catalog import (
    CompilationCatalog,
    CompilationOption,
    get_compilation_catalog,
)
from domain.compilation_plan import CompilationObject
from domain.graph import ModelGraph
from domain.metric_formula import MetricFormulaError, evaluate_metric_formula, parse_metric_formula
from validation.output_resolution import output_selectors, resolved_outputs


###############################################################################
def option(value: Any) -> Any:
    """Convert the catalog's explicit ``none`` option into Keras ``None``."""

    return None if value in (None, "", "none") else value

###############################################################################
def layer_option(
    parameters: dict[str, Any],
    key: str,
    graph: ModelGraph,
    fallback: Any,
) -> Any:
    """Resolve a layer option, applying model defaults only when unset."""

    if key in parameters:
        return option(parameters[key])
    configuration = graph.model_configuration.keras
    if "initializer" in key:
        return option(configuration.compilation.default_initializer.id)
    if "regularizer" in key:
        return option(
            configuration.compilation.default_regularizer.id
            if configuration.compilation.default_regularizer
            else "none"
        )
    if "constraint" in key:
        return option(
            configuration.compilation.default_constraint.id
            if configuration.compilation.default_constraint
            else "none"
        )
    return option(fallback)

###############################################################################
def prepare_runtime(graph: ModelGraph) -> None:
    if graph.model_configuration.keras.random_seed is not None:
        keras.utils.set_random_seed(graph.model_configuration.keras.random_seed)
    keras.mixed_precision.set_global_policy(
        "mixed_float16" if graph.model_configuration.keras.mixed_precision else "float32"
    )


###############################################################################
class PredictionAggregateMetric(keras.metrics.Metric):
    """Stateful metric whose input is explicitly the prediction tensor."""

    # -------------------------------------------------------------------------
    def __init__(self, mode: str, name: str | None = None, **kwargs: Any) -> None:
        super().__init__(name=name or f"prediction_{mode}", **kwargs)
        if mode not in {"mean", "sum"}:
            raise ValueError(f"Unsupported prediction aggregate mode: {mode}")
        self.mode = mode
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    # -------------------------------------------------------------------------
    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Any = None
    ) -> None:
        del y_true
        values = keras.ops.cast(y_pred, self.dtype)
        if sample_weight is not None:
            values = values * keras.ops.cast(sample_weight, values.dtype)
        self.total.assign_add(keras.ops.sum(values))
        if self.mode == "mean":
            self.count.assign_add(keras.ops.cast(keras.ops.size(values), self.dtype))
        else:
            self.count.assign_add(keras.ops.cast(1, self.dtype))

    # -------------------------------------------------------------------------
    def result(self) -> Any:
        if self.mode == "sum":
            return self.total
        return self.total / keras.ops.maximum(self.count, keras.ops.cast(1, self.dtype))

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "mode": self.mode}

###############################################################################
def _output_names(graph: ModelGraph) -> list[tuple[str, set[str]]]:
    return [
        (
            output.name,
            set(output_selectors(output)),
        )
        for output in resolved_outputs(graph, get_node_catalog())
        if output.node is not None
    ]

###############################################################################
def _configured_values(
    graph: ModelGraph,
    configuration: dict[str, CompilationObject],
    *,
    default: Any,
) -> dict[str, Any]:
    outputs = _output_names(graph)
    if not configuration:
        return {}
    resolved: dict[str, Any] = {}
    for output_name, selectors in outputs:
        value = configuration.get("__default__", default)
        for selector in selectors:
            if selector in configuration:
                value = configuration[selector]
                break
        resolved[output_name] = value
    return resolved

###############################################################################
def _compile_value(values: dict[str, Any], output_count: int) -> Any:
    if not values:
        return None
    if output_count == 1:
        return next(iter(values.values()))
    return list(values.values())

###############################################################################
def _catalog_option(category: str, identifier: str, catalog: CompilationCatalog) -> CompilationOption:
    options = getattr(catalog, category)
    option = next((item for item in options if item.id == identifier), None)
    if option is None:
        raise ValueError(f"No Keras catalog option registered for {category}.{identifier}")
    implementation = option.implementations.get("keras")
    if implementation is None or not implementation.supported:
        raise ValueError(f"No Keras implementation registered for {category}.{identifier}")
    return option

###############################################################################
def _optimizer(value: CompilationObject, catalog: CompilationCatalog) -> keras.optimizers.Optimizer:
    option = _catalog_option("optimizers", value.id, catalog)
    implementation = option.implementations["keras"]
    optimizer_type = getattr(keras.optimizers, implementation.symbol or "")
    accepted_parameters = {parameter.id for parameter in option.parameters}
    kwargs = {
        implementation.parameter_mapping.get(key, key): parameter_value
        for key, parameter_value in value.parameters.items()
        if key in accepted_parameters
    }
    return optimizer_type(**kwargs)

###############################################################################
def _custom_formula(option: CompilationOption) -> Any:
    try:
        tree = parse_metric_formula(option.formula or "")
    except MetricFormulaError as exc:
        raise ValueError(f"Invalid custom formula for {option.id}: {exc}") from exc
    operations = {
        "abs": keras.ops.abs,
        "square": keras.ops.square,
        "sqrt": keras.ops.sqrt,
        "log": keras.ops.log,
        "exp": keras.ops.exp,
        "mean": keras.ops.mean,
        "sum": keras.ops.sum,
    }

    def custom(y_true: Any, y_pred: Any) -> Any:
        return evaluate_metric_formula(tree, y_true, y_pred, operations)

    custom.__name__ = option.id
    return custom

###############################################################################
def _compile_option(
    category: str, value: CompilationObject, catalog: CompilationCatalog
) -> Any:
    option = _catalog_option(category, value.id, catalog)
    if option.origin == "custom":
        return _custom_formula(option)
    implementation = option.implementations["keras"]
    if implementation.adapter == "prediction_mean":
        return PredictionAggregateMetric("mean", name=option.id)
    if implementation.adapter == "prediction_sum":
        return PredictionAggregateMetric("sum", name=option.id)
    if not implementation.symbol:
        raise ValueError(f"No Keras symbol registered for {category}.{value.id}")
    namespace = keras.losses if category == "losses" else keras.metrics
    factory = getattr(namespace, implementation.symbol or "")
    accepted_parameters = {parameter.id for parameter in option.parameters}
    kwargs = {
        implementation.parameter_mapping.get(key, key): parameter_value
        for key, parameter_value in value.parameters.items()
        if key in accepted_parameters
    }
    return factory(**kwargs)

###############################################################################
def _resolve_compilation_values(values: dict[str, Any], category: str, catalog: CompilationCatalog) -> dict[str, Any]:
    return {
        output_id: (
            _compile_option(category, value, catalog)
            if category == "losses" and isinstance(value, CompilationObject)
            else [_compile_option(category, item, catalog) for item in value]
            if category == "metrics" and isinstance(value, list)
            else value
        )
        for output_id, value in values.items()
    }

###############################################################################
def compile_model(model: keras.Model, graph: ModelGraph, *, compilation_catalog: CompilationCatalog | None = None) -> keras.Model:
    compilation_catalog = compilation_catalog or get_compilation_catalog()
    configuration = graph.model_configuration.keras
    compilation = configuration.compilation
    losses = _configured_values(graph, compilation.losses, default=None)
    metrics = _configured_values(graph, compilation.metrics, default=[])
    losses = _resolve_compilation_values(losses, "losses", compilation_catalog)
    metrics = _resolve_compilation_values(metrics, "metrics", compilation_catalog)
    model.compile(
        optimizer=_optimizer(compilation.optimizer, compilation_catalog),
        loss=_compile_value(losses, len(_output_names(graph))),
        metrics=_compile_value(metrics, len(_output_names(graph))),
    )
    return model
