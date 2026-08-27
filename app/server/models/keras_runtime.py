from __future__ import annotations

from typing import Any

import keras

from domain.graph import ModelGraph


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
        return option(configuration.default_initializer)
    if "regularizer" in key:
        return option(configuration.default_regularizer)
    if "constraint" in key:
        return option(configuration.default_constraint)
    return option(fallback)

###############################################################################
def prepare_runtime(graph: ModelGraph) -> None:
    if graph.model_configuration.keras.random_seed is not None:
        keras.utils.set_random_seed(graph.model_configuration.keras.random_seed)
    keras.mixed_precision.set_global_policy(
        "mixed_float16" if graph.model_configuration.keras.mixed_precision else "float32"
    )

###############################################################################
def _output_names(graph: ModelGraph) -> list[tuple[str, set[str]]]:
    return [
        (
            node.id,
            {node.id, node.label or node.id},
        )
        for node in graph.nodes
        if node.definition_id.startswith("output.")
    ]

###############################################################################
def _configured_values(
    graph: ModelGraph,
    configuration: dict[str, Any],
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
    return values

###############################################################################
def _optimizer(name: str, parameters: dict[str, Any]) -> keras.optimizers.Optimizer:
    optimizer_types = {
        "adam": keras.optimizers.Adam,
        "adamw": keras.optimizers.AdamW,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
    }
    accepted_parameters = {
        "adam": {"learning_rate", "beta_1", "beta_2", "epsilon"},
        "adamw": {"learning_rate", "weight_decay", "beta_1", "beta_2", "epsilon"},
        "sgd": {"learning_rate", "momentum", "nesterov"},
        "rmsprop": {"learning_rate", "rho", "momentum", "epsilon", "centered"},
    }[name]
    kwargs = {key: value for key, value in parameters.items() if key in accepted_parameters}
    return optimizer_types[name](**kwargs)

###############################################################################
def compile_model(model: keras.Model, graph: ModelGraph) -> keras.Model:
    configuration = graph.model_configuration.keras
    losses = _configured_values(graph, configuration.losses, default=None)
    metrics = _configured_values(graph, configuration.metrics, default=[])
    model.compile(
        optimizer=_optimizer(configuration.optimizer, configuration.optimizer_parameters),
        loss=_compile_value(losses, len(_output_names(graph))),
        metrics=_compile_value(metrics, len(_output_names(graph))),
    )
    return model
