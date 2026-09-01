from __future__ import annotations

import math
from typing import Any, cast

from common.catalogs.loader import get_node_catalog
from domain.compilation_catalog import (
    CompilationCatalog,
    get_compilation_catalog,
    get_compilation_catalog_for_implementation,
)
from domain.diagnostics import ValidationDiagnostic
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import ModelGraph
from validation.output_resolution import output_selectors, resolved_outputs


###############################################################################
def _error(
    code: str,
    message: str,
    *,
    node_id: str | None = None,
    parameter_id: str | None = None,
    details: dict[str, object] | None = None,
) -> ValidationDiagnostic:
    return ValidationDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        message=message,
        node_id=node_id,
        parameter_id=parameter_id,
        details=details or {},
    )


###############################################################################
def _output_selectors(graph: ModelGraph) -> set[str]:
    return {
        selector
        for output in resolved_outputs(graph, get_node_catalog())
        for selector in output_selectors(output)
    }


###############################################################################
def _selected_value(configuration: dict[str, Any], output: Any, default: Any) -> Any:
    value = configuration.get("__default__", default)
    selectors = output_selectors(output) if hasattr(output, "binding") else (
        output.id,
        output.label,
    )
    for selector in selectors:
        if selector and selector in configuration:
            value = configuration[selector]
            break
    return value


###############################################################################
def _validate_optimizer(
    configuration: Any, catalog: CompilationCatalog, runtime_label: str
) -> list[ValidationDiagnostic]:
    diagnostics: list[ValidationDiagnostic] = []
    optimizer_by_id = {item.id: item for item in catalog.optimizers}
    optimizer = optimizer_by_id.get(configuration.optimizer)
    if optimizer is None:
        diagnostics.append(
            _error(
                "invalid_compile_optimizer",
                f"Optimizer '{configuration.optimizer}' is not supported by {runtime_label}",
            )
        )
        return diagnostics
    allowed_parameters = {parameter.id: parameter for parameter in optimizer.parameters}
    unknown_parameters = set(configuration.optimizer_parameters) - set(allowed_parameters)
    if unknown_parameters:
        diagnostics.append(
            _error(
                "unsupported_optimizer_parameter",
                f"Optimizer '{configuration.optimizer}' does not support: {', '.join(sorted(unknown_parameters))}",
            )
        )
    for parameter_id, value in configuration.optimizer_parameters.items():
        parameter = allowed_parameters.get(parameter_id)
        if parameter is None:
            continue
        if parameter.type == "boolean":
            if not isinstance(value, bool):
                diagnostics.append(
                    _error(
                        "invalid_optimizer_parameter",
                        f"optimizer_parameters.{parameter_id} must be a boolean",
                    )
                )
            continue
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            diagnostics.append(
                _error(
                    "invalid_optimizer_parameter",
                    f"optimizer_parameters.{parameter_id} must be a finite number",
                )
            )
            continue
        minimum = parameter.constraints.get("min")
        maximum = parameter.constraints.get("max")
        if isinstance(minimum, int | float) and value < minimum:
            diagnostics.append(
                _error(
                    "invalid_optimizer_parameter",
                    f"optimizer_parameters.{parameter_id} is outside its supported range",
                )
            )
        if isinstance(maximum, int | float) and value > maximum:
            diagnostics.append(
                _error(
                    "invalid_optimizer_parameter",
                    f"optimizer_parameters.{parameter_id} is outside its supported range",
                )
            )
        if parameter_id == "learning_rate" and value <= 0:
            diagnostics.append(
                _error(
                    "invalid_optimizer_learning_rate",
                    "optimizer_parameters.learning_rate must be a finite positive number",
                )
            )
    return diagnostics


###############################################################################
def _validate_mappings(
    graph: ModelGraph,
    configuration: Any,
    catalog: CompilationCatalog,
    runtime_label: str,
) -> list[ValidationDiagnostic]:
    diagnostics = _validate_optimizer(configuration, catalog, runtime_label)
    supported_losses = {item.id for item in catalog.losses}
    supported_metrics = {item.id for item in catalog.metrics}
    selectors = _output_selectors(graph)
    for output_selector, loss in configuration.losses.items():
        if output_selector != "__default__" and output_selector not in selectors:
            diagnostics.append(
                _error(
                    "unknown_compile_output",
                    f"Loss configuration references unknown output '{output_selector}'",
                )
            )
        if loss not in supported_losses:
            diagnostics.append(
                _error(
                    "invalid_compile_loss",
                    f"Loss '{loss}' is not supported by {runtime_label}",
                )
            )
    for output_selector, metric_values in configuration.metrics.items():
        if output_selector != "__default__" and output_selector not in selectors:
            diagnostics.append(
                _error(
                    "unknown_compile_output",
                    f"Metric configuration references unknown output '{output_selector}'",
                )
            )
        for metric in metric_values:
            if metric not in supported_metrics:
                diagnostics.append(
                    _error(
                        "invalid_compile_metric",
                        f"Metric '{metric}' is not supported by {runtime_label}",
                    )
                )
    return diagnostics


###############################################################################
def _terminal_head(graph: ModelGraph, output_id: str) -> tuple[str | None, int | None]:
    """Find the final activation and class width for one selected output."""

    nodes = {node.id: node for node in graph.nodes}
    incoming = {
        edge.target_node_id: edge.source_node_id
        for edge in graph.edges
        if edge.target_port_id == "input"
    }
    current = nodes.get(output_id)
    visited: set[str] = set()
    while current is not None and current.id not in visited:
        visited.add(current.id)
        node = current
        if node.definition_id.startswith("output."):
            source_id = incoming.get(node.id)
            current = nodes.get(source_id) if source_id is not None else None
            continue
        if node.definition_id == "layer.activation":
            return str(node.parameters.get("activation", "linear")), None
        if node.definition_id in {
            "layer.dense",
            "conv.conv1d",
            "conv.conv2d",
            "layer.embedding",
        }:
            width_value = node.parameters.get("units", node.parameters.get("filters"))
            width = (
                width_value
                if isinstance(width_value, int) and not isinstance(width_value, bool)
                else None
            )
            return str(node.parameters.get("activation", "linear")), width
        source_id = incoming.get(node.id)
        current = nodes.get(source_id) if source_id is not None else None
    return None, None


###############################################################################
def _output_task(output: Any, width: int | None) -> str | None:
    if output.definition_id == "output.regression":
        return "regression"
    if output.definition_id == "output.classification":
        return "binary_classification" if width == 1 else "multiclass_classification"
    return None


###############################################################################
def _validate_semantics(
    graph: ModelGraph,
    configuration: Any,
    catalog: CompilationCatalog,
) -> list[ValidationDiagnostic]:
    diagnostics: list[ValidationDiagnostic] = []
    losses = {item.id: item for item in catalog.losses}
    metrics = {item.id: item for item in catalog.metrics}
    output_ids = {
        selector
        for selector in _output_selectors(graph)
        if any(node.id == selector for node in graph.nodes)
    }
    output_nodes = [node for node in graph.nodes if node.id in output_ids]
    for output in output_nodes:
        activation, width = _terminal_head(graph, output.id)
        task = _output_task(output, width)
        if task is None:
            continue
        if task == "binary_classification":
            if width is not None and width != 1:
                diagnostics.append(
                    _error(
                        "invalid_binary_output_shape",
                        "Binary classification outputs must have exactly one prediction value",
                        node_id=output.id,
                    )
                )
            expected_activation = "sigmoid"
        elif task == "multiclass_classification":
            if width is not None and width < 2:
                diagnostics.append(
                    _error(
                        "invalid_multiclass_output_shape",
                        "Multiclass classification outputs must expose at least two classes",
                        node_id=output.id,
                    )
                )
            expected_activation = "softmax"
        else:
            expected_activation = "linear"
        if activation != expected_activation and (
            activation is not None or expected_activation != "linear"
        ):
            diagnostics.append(
                _error(
                    "invalid_output_activation",
                    f"{output.definition_id} requires a final {expected_activation} activation; received {activation or 'none'}",
                    node_id=output.id,
                    details={"expected": expected_activation, "received": activation},
                )
            )

        selected_loss = _selected_value(configuration.losses, output, None)
        loss = losses.get(selected_loss) if isinstance(selected_loss, str) else None
        if loss is not None:
            if task not in loss.supported_tasks:
                diagnostics.append(
                    _error(
                        "incompatible_compile_loss",
                        f"Loss '{loss.id}' is incompatible with {output.definition_id}",
                        node_id=output.id,
                        details={
                            "task": task,
                            "supported_tasks": loss.supported_tasks,
                            "target_contract": loss.target_contract,
                        },
                    )
                )
            if loss.activation_contract and activation not in loss.activation_contract:
                diagnostics.append(
                    _error(
                        "invalid_loss_activation",
                        f"Loss '{loss.id}' requires one of: {', '.join(loss.activation_contract)}",
                        node_id=output.id,
                        details={
                            "loss": loss.id,
                            "expected": loss.activation_contract,
                            "received": activation,
                        },
                    )
                )

        selected_metrics = _selected_value(configuration.metrics, output, [])
        if isinstance(selected_metrics, list):
            for metric_id in cast(list[Any], selected_metrics):
                if not isinstance(metric_id, str):
                    diagnostics.append(
                        _error(
                            "invalid_compile_metric",
                            "Metrics must be selected by string identifier",
                            node_id=output.id,
                        )
                    )
                    continue
                metric = metrics.get(metric_id)
                if metric is not None and task not in metric.supported_tasks:
                    diagnostics.append(
                        _error(
                            "incompatible_compile_metric",
                            f"Metric '{metric.id}' is incompatible with {output.definition_id}",
                            node_id=output.id,
                            details={
                                "task": task,
                                "supported_tasks": metric.supported_tasks,
                                "target_contract": metric.target_contract,
                            },
                        )
                    )
    return diagnostics


###############################################################################
def validate_model_configuration(
    graph: ModelGraph,
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
    catalog: CompilationCatalog | None = None,
) -> list[ValidationDiagnostic]:
    """Validate the selected backend's canonical compilation configuration."""

    catalog = catalog or get_compilation_catalog()
    if not keras_enabled:
        configuration = graph.model_configuration.native
        if execution_backend is ExecutionBackend.TENSORFLOW:
            if not isinstance(configuration.tensorflow.jit_compile, bool):
                return [
                    _error("invalid_native_compilation", "TensorFlow jit_compile must be boolean")
                ]
            return []
        if execution_backend is ExecutionBackend.PYTORCH:
            native = configuration.pytorch
            native_catalog = get_compilation_catalog_for_implementation(catalog, "pytorch")
            diagnostics = _validate_mappings(
                graph, native, native_catalog, "the native PyTorch runtime"
            )
            if native.mode not in {"default", "reduce-overhead", "max-autotune"}:
                diagnostics.append(
                    _error("invalid_native_compilation", "PyTorch compile mode is unsupported")
                )
            for parameter_id, value in {
                "compile_enabled": native.compile_enabled,
                "fullgraph": native.fullgraph,
                "dynamic": native.dynamic,
            }.items():
                if not isinstance(value, bool):
                    diagnostics.append(
                        _error(
                            "invalid_native_compilation", f"PyTorch {parameter_id} must be boolean"
                        )
                    )
            diagnostics.extend(_validate_semantics(graph, native, native_catalog))
            return diagnostics
        if not isinstance(configuration.jax.jit_enabled, bool):
            return [_error("invalid_native_compilation", "JAX jit_enabled must be boolean")]
        return []

    configuration = graph.model_configuration.keras
    diagnostics = _validate_mappings(graph, configuration, catalog, "the Keras runtime")
    supported_initializers = frozenset(catalog.initializers)
    supported_regularizers = frozenset(catalog.regularizers)
    supported_constraints = frozenset(catalog.constraints)
    if configuration.default_initializer not in supported_initializers:
        diagnostics.append(
            _error(
                "invalid_default_initializer",
                f"Initializer '{configuration.default_initializer}' is not supported",
            )
        )
    if configuration.default_regularizer not in supported_regularizers:
        diagnostics.append(
            _error(
                "invalid_default_regularizer",
                f"Regularizer '{configuration.default_regularizer}' is not supported",
            )
        )
    if configuration.default_constraint not in supported_constraints:
        diagnostics.append(
            _error(
                "invalid_default_constraint",
                f"Constraint '{configuration.default_constraint}' is not supported",
            )
        )
    diagnostics.extend(_validate_semantics(graph, configuration, catalog))
    return diagnostics
