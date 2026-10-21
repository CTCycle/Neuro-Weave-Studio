# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

from common.catalogs.loader import get_node_catalog
from domain.compilation_catalog import (
    CompilationCatalog,
    get_compilation_catalog,
    get_compilation_catalog_for_implementation,
)
from domain.compilation_plan import CompilationObject
from domain.diagnostics import ValidationDiagnostic
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import ModelGraph
from validation.output_resolution import ResolvedOutput, output_selectors, resolved_outputs

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
def _output_selectors(outputs: Sequence[ResolvedOutput]) -> set[str]:
    return {
        selector
        for output in outputs
        for selector in output_selectors(output)
    }

###############################################################################
def _selected_value(
    configuration: dict[str, Any], output: ResolvedOutput, default: Any
) -> Any:
    value = configuration.get("__default__", default)
    for selector in output_selectors(output):
        if selector and selector in configuration:
            value = configuration[selector]
            break
    return value

###############################################################################
def _output_details(output: ResolvedOutput, task: str | None = None) -> dict[str, object]:
    details: dict[str, object] = {
        "output_id": f"{output.binding.node_id}:{output.binding.port_id}",
        "output_name": output.name,
        "port_id": output.binding.port_id,
        "selectors": list(output_selectors(output)),
    }
    if task is not None:
        details["task"] = task
    return details

###############################################################################
def _matching_outputs(
    output_selector: str, outputs: Sequence[ResolvedOutput]
) -> list[ResolvedOutput]:
    if output_selector == "__default__":
        return list(outputs)
    return [output for output in outputs if output_selector in output_selectors(output)]

###############################################################################
def _validate_optimizer(
    configuration: Any, catalog: CompilationCatalog, runtime_label: str
) -> list[ValidationDiagnostic]:
    diagnostics: list[ValidationDiagnostic] = []
    optimizer_by_id = {item.id: item for item in catalog.optimizers}
    compilation = configuration.compilation
    optimizer_value = compilation.optimizer
    optimizer = optimizer_by_id.get(optimizer_value.id)
    if optimizer is None:
        diagnostics.append(
            _error(
                "invalid_compile_optimizer",
                f"Optimizer '{optimizer_value.id}' is not supported by {runtime_label}",
            )
        )
        return diagnostics
    allowed_parameters = {parameter.id: parameter for parameter in optimizer.parameters}
    unknown_parameters = set(optimizer_value.parameters) - set(allowed_parameters)
    if unknown_parameters:
        diagnostics.append(
            _error(
                "unsupported_optimizer_parameter",
                f"Optimizer '{optimizer_value.id}' does not support: {', '.join(sorted(unknown_parameters))}",
            )
        )
    for parameter_id, value in optimizer_value.parameters.items():
        parameter = allowed_parameters.get(parameter_id)
        if parameter is None:
            continue
        if parameter.type == "boolean":
            if not isinstance(value, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
                diagnostics.append(
                    _error(
                        "invalid_optimizer_parameter",
                        f"optimizer.parameters.{parameter_id} must be a boolean",
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
                    f"optimizer.parameters.{parameter_id} must be a finite number",
                )
            )
            continue
        minimum = parameter.constraints.get("min")
        maximum = parameter.constraints.get("max")
        if isinstance(minimum, int | float) and value < minimum:
            diagnostics.append(
                _error(
                    "invalid_optimizer_parameter",
                    f"optimizer.parameters.{parameter_id} is outside its supported range",
                )
            )
        if isinstance(maximum, int | float) and value > maximum:
            diagnostics.append(
                _error(
                    "invalid_optimizer_parameter",
                    f"optimizer.parameters.{parameter_id} is outside its supported range",
                )
            )
        if parameter_id == "learning_rate" and value <= 0:
            diagnostics.append(
                _error(
                    "invalid_optimizer_learning_rate",
                    "optimizer.parameters.learning_rate must be a finite positive number",
                )
            )
    return diagnostics

###############################################################################
def _validate_mappings(
    configuration: Any,
    catalog: CompilationCatalog,
    runtime_label: str,
    outputs: Sequence[ResolvedOutput],
) -> list[ValidationDiagnostic]:
    diagnostics = _validate_optimizer(configuration, catalog, runtime_label)
    supported_losses = {item.id for item in catalog.losses}
    supported_metrics = {item.id for item in catalog.metrics}
    selectors = _output_selectors(outputs)
    compilation = configuration.compilation
    for output_selector, loss in compilation.losses.items():
        targets = _matching_outputs(output_selector, outputs)
        if output_selector != "__default__" and not targets:
            diagnostics.append(
                _error(
                    "unknown_compile_output",
                    f"Loss configuration references unknown output '{output_selector}'",
                    parameter_id="losses",
                    details={
                        "output_selector": output_selector,
                        "available_selectors": sorted(selectors),
                    },
                )
            )
        elif output_selector != "__default__" and len(targets) > 1:
            diagnostics.append(
                _error(
                    "ambiguous_compile_output",
                    f"Loss configuration selector '{output_selector}' matches multiple outputs",
                    parameter_id="losses",
                    details={
                        "output_selector": output_selector,
                        "outputs": [item.name for item in targets],
                    },
                )
            )
        if loss.id not in supported_losses:
            for target in targets or [None]:
                diagnostics.append(
                    _error(
                        "invalid_compile_loss",
                        f"Loss '{loss.id}' is not supported by {runtime_label}"
                        + (f" for output '{target.name}'" if target is not None else ""),
                        node_id=target.binding.node_id if target is not None else None,
                        parameter_id="losses",
                        details=(
                            _output_details(target)
                            if target is not None
                            else {"output_selector": output_selector}
                        ),
                    )
                )
    for output_selector, metric_values in compilation.metrics.items():
        targets = _matching_outputs(output_selector, outputs)
        if output_selector != "__default__" and not targets:
            diagnostics.append(
                _error(
                    "unknown_compile_output",
                    f"Metric configuration references unknown output '{output_selector}'",
                    parameter_id="metrics",
                    details={
                        "output_selector": output_selector,
                        "available_selectors": sorted(selectors),
                    },
                )
            )
        elif output_selector != "__default__" and len(targets) > 1:
            diagnostics.append(
                _error(
                    "ambiguous_compile_output",
                    f"Metric configuration selector '{output_selector}' matches multiple outputs",
                    parameter_id="metrics",
                    details={
                        "output_selector": output_selector,
                        "outputs": [item.name for item in targets],
                    },
                )
            )
        for metric in metric_values:
            if metric.id not in supported_metrics:
                for target in targets or [None]:
                    diagnostics.append(
                        _error(
                            "invalid_compile_metric",
                            f"Metric '{metric.id}' is not supported by {runtime_label}"
                            + (f" for output '{target.name}'" if target is not None else ""),
                            node_id=target.binding.node_id if target is not None else None,
                            parameter_id="metrics",
                            details=(
                                _output_details(target)
                                if target is not None
                                else {"output_selector": output_selector}
                            ),
                        )
                    )
    return diagnostics

###############################################################################
def _terminal_head(graph: ModelGraph, output: ResolvedOutput) -> tuple[str | None, int | None]:
    """Find the final activation and class width for one selected output."""

    nodes = {node.id: node for node in graph.nodes}
    incoming = {
        edge.target_node_id: edge.source_node_id
        for edge in graph.edges
        if edge.target_port_id == "input"
    }
    current = output.node
    visited: set[str] = set()
    while current is not None and current.id not in visited:
        visited.add(current.id)
        node = current
        if node is output.node and output.manifest is not None and output.manifest.semantic_role == "output_adapter":
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
def _output_task(output: ResolvedOutput, width: int | None) -> str | None:
    definition_id = output.manifest.id if output.manifest is not None else None
    if definition_id == "output.regression":
        return "regression"
    if definition_id == "output.classification" and width is not None:
        return "binary_classification" if width == 1 else "multiclass_classification"
    return None

###############################################################################
def _validate_semantics(
    graph: ModelGraph,
    configuration: Any,
    catalog: CompilationCatalog,
    outputs: Sequence[ResolvedOutput],
) -> list[ValidationDiagnostic]:
    diagnostics: list[ValidationDiagnostic] = []
    losses = {item.id: item for item in catalog.losses}
    metrics = {item.id: item for item in catalog.metrics}
    for output in outputs:
        if output.node is None or output.manifest is None or output.port is None:
            continue
        activation, width = _terminal_head(graph, output)
        task = _output_task(output, width)
        if task is None:
            continue
        output_details = _output_details(output, task)
        node_id = output.binding.node_id
        if task == "binary_classification":
            if width is not None and width != 1:
                diagnostics.append(
                    _error(
                        "invalid_binary_output_shape",
                        "Binary classification outputs must have exactly one prediction value",
                        node_id=node_id,
                        details=output_details,
                    )
                )
            expected_activation = "sigmoid"
        elif task == "multiclass_classification":
            if width is not None and width < 2:
                diagnostics.append(
                    _error(
                        "invalid_multiclass_output_shape",
                        "Multiclass classification outputs must expose at least two classes",
                        node_id=node_id,
                        details=output_details,
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
                    f"{output.manifest.id} requires a final {expected_activation} activation; received {activation or 'none'}",
                    node_id=node_id,
                    details={**output_details, "expected": expected_activation, "received": activation},
                )
            )

        selected_loss = _selected_value(configuration.compilation.losses, output, None)
        loss = (
            losses.get(selected_loss.id)
            if isinstance(selected_loss, CompilationObject)
            else None
        )
        if loss is not None:
            if task not in loss.supported_tasks:
                diagnostics.append(
                    _error(
                        "incompatible_compile_loss",
                        f"Loss '{loss.id}' is incompatible with output '{output.name}'",
                        node_id=node_id,
                        parameter_id="losses",
                        details={
                            **output_details,
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
                        node_id=node_id,
                        parameter_id="losses",
                        details={
                            **output_details,
                            "loss": loss.id,
                            "expected": loss.activation_contract,
                            "received": activation,
                        },
                    )
                )

        selected_metrics = _selected_value(configuration.compilation.metrics, output, [])
        if isinstance(selected_metrics, list):
            for metric_value in cast(list[Any], selected_metrics):
                if not isinstance(metric_value, CompilationObject):
                    diagnostics.append(
                        _error(
                            "invalid_compile_metric",
                            "Metrics must be selected by string identifier",
                            node_id=node_id,
                            parameter_id="metrics",
                            details=output_details,
                        )
                    )
                    continue
                metric = metrics.get(metric_value.id)
                if metric is not None and task not in metric.supported_tasks:
                    diagnostics.append(
                        _error(
                            "incompatible_compile_metric",
                            f"Metric '{metric.id}' is incompatible with output '{output.name}'",
                            node_id=node_id,
                            parameter_id="metrics",
                            details={
                                **output_details,
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
    outputs: Sequence[ResolvedOutput] | None = None,
) -> list[ValidationDiagnostic]:
    """Validate the selected backend's canonical compilation configuration."""

    catalog = catalog or get_compilation_catalog()
    resolved = (
        resolved_outputs(graph, get_node_catalog()) if outputs is None else list(outputs)
    )
    if not keras_enabled:
        configuration = graph.model_configuration.native
        if execution_backend is ExecutionBackend.TENSORFLOW:
            if not isinstance(configuration.tensorflow.compiler.parameters.get("jit_compile", False), bool):  # pyright: ignore[reportUnnecessaryIsInstance]
                return [
                    _error("invalid_native_compilation", "TensorFlow jit_compile must be boolean")
                ]
            return []
        if execution_backend is ExecutionBackend.PYTORCH:
            native = configuration.pytorch
            native_catalog = get_compilation_catalog_for_implementation(catalog, "pytorch")
            diagnostics = _validate_mappings(
                native, native_catalog, "the native PyTorch runtime", resolved
            )
            mode = native.compiler.parameters.get("mode", "default")
            if mode not in {"default", "reduce-overhead", "max-autotune"}:
                diagnostics.append(
                    _error("invalid_native_compilation", "PyTorch compile mode is unsupported")
                )
            for parameter_id, value in {
                "compile_enabled": native.compiler.parameters.get("enabled", False),
                "fullgraph": native.compiler.parameters.get("fullgraph", False),
                "dynamic": native.compiler.parameters.get("dynamic", False),
            }.items():
                if not isinstance(value, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
                    diagnostics.append(
                        _error(
                            "invalid_native_compilation", f"PyTorch {parameter_id} must be boolean"
                        )
                    )
            diagnostics.extend(_validate_semantics(graph, native, native_catalog, resolved))
            return diagnostics
        if not isinstance(configuration.jax.compiler.parameters.get("enabled", False), bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            return [_error("invalid_native_compilation", "JAX jit_enabled must be boolean")]
        return []

    configuration = graph.model_configuration.keras
    diagnostics = _validate_mappings(configuration, catalog, "the Keras runtime", resolved)
    supported_initializers = frozenset(catalog.initializers)
    supported_regularizers = frozenset(catalog.regularizers)
    supported_constraints = frozenset(catalog.constraints)
    compilation = configuration.compilation
    if compilation.default_initializer.id not in supported_initializers:
        diagnostics.append(
            _error(
                "invalid_default_initializer",
                f"Initializer '{compilation.default_initializer.id}' is not supported",
            )
        )
    if compilation.default_regularizer and compilation.default_regularizer.id not in supported_regularizers:
        diagnostics.append(
            _error(
                "invalid_default_regularizer",
                f"Regularizer '{compilation.default_regularizer.id}' is not supported",
            )
        )
    if compilation.default_constraint and compilation.default_constraint.id not in supported_constraints:
        diagnostics.append(
            _error(
                "invalid_default_constraint",
                f"Constraint '{compilation.default_constraint.id}' is not supported",
            )
        )
    diagnostics.extend(_validate_semantics(graph, configuration, catalog, resolved))
    return diagnostics
