from __future__ import annotations

import math

from domain.compilation_catalog import CompilationCatalog, get_compilation_catalog
from domain.diagnostics import ValidationDiagnostic
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import ModelGraph

###############################################################################
def _error(code: str, message: str) -> ValidationDiagnostic:
    return ValidationDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        message=message,
    )

###############################################################################
def _output_selectors(graph: ModelGraph) -> set[str]:
    return {
        selector
        for node in graph.nodes
        if node.definition_id.startswith("output.")
        for selector in (node.id, node.label or node.id)
    }

###############################################################################
def validate_model_configuration(
    graph: ModelGraph,
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
    catalog: CompilationCatalog | None = None,
) -> list[ValidationDiagnostic]:
    """Validate the selected backend's canonical compilation configuration."""

    catalog = catalog or get_compilation_catalog()
    supported_optimizers = frozenset(item.id for item in catalog.optimizers)
    supported_losses = frozenset(item.id for item in catalog.losses)
    supported_metrics = frozenset(item.id for item in catalog.metrics)
    optimizer_parameters = {
        item.id: {parameter.id: parameter for parameter in item.parameters}
        for item in catalog.optimizers
    }
    supported_initializers = frozenset(catalog.initializers)
    supported_regularizers = frozenset(catalog.regularizers)
    supported_constraints = frozenset(catalog.constraints)

    if not keras_enabled:
        configuration = graph.model_configuration.native
        if execution_backend is ExecutionBackend.TENSORFLOW:
            if not isinstance(configuration.tensorflow.jit_compile, bool):
                return [_error("invalid_native_compilation", "TensorFlow jit_compile must be boolean")]
            return []
        if execution_backend is ExecutionBackend.PYTORCH:
            if configuration.pytorch.mode not in {"default", "reduce-overhead", "max-autotune"}:
                return [_error("invalid_native_compilation", "PyTorch compile mode is unsupported")]
            return []
        if not isinstance(configuration.jax.jit_enabled, bool):
            return [_error("invalid_native_compilation", "JAX jit_enabled must be boolean")]
        return []

    configuration = graph.model_configuration.keras
    diagnostics: list[ValidationDiagnostic] = []
    if configuration.optimizer not in supported_optimizers:
        diagnostics.append(
            _error(
                "invalid_compile_optimizer",
                f"Optimizer '{configuration.optimizer}' is not supported by the Keras runtime",
            )
        )

    allowed_parameters = optimizer_parameters.get(configuration.optimizer, {})
    unknown_optimizer_parameters = set(configuration.optimizer_parameters) - set(allowed_parameters)
    if unknown_optimizer_parameters:
        diagnostics.append(
            _error(
                "unsupported_optimizer_parameter",
                f"Optimizer '{configuration.optimizer}' does not support: {', '.join(sorted(unknown_optimizer_parameters))}",
            )
        )
    for parameter_id, value in configuration.optimizer_parameters.items():
        parameter = allowed_parameters.get(parameter_id)
        if parameter is None:
            continue
        if parameter.type == "boolean":
            if not isinstance(value, bool):
                diagnostics.append(_error("invalid_optimizer_parameter", f"optimizer_parameters.{parameter_id} must be a boolean"))
            continue
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
            diagnostics.append(_error("invalid_optimizer_parameter", f"optimizer_parameters.{parameter_id} must be a finite number"))
            continue
        minimum = parameter.constraints.get("min")
        maximum = parameter.constraints.get("max")
        if (isinstance(minimum, int | float) and value < minimum) or (isinstance(maximum, int | float) and value > maximum):
            diagnostics.append(_error("invalid_optimizer_parameter", f"optimizer_parameters.{parameter_id} is outside its supported range"))
        if parameter_id == "learning_rate" and value <= 0:
            diagnostics.append(_error("invalid_optimizer_learning_rate", "optimizer_parameters.learning_rate must be a finite positive number"))

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
                    f"Loss '{loss}' is not supported by the Keras runtime",
                )
            )
    for output_selector, metrics in configuration.metrics.items():
        if output_selector != "__default__" and output_selector not in selectors:
            diagnostics.append(
                _error(
                    "unknown_compile_output",
                    f"Metric configuration references unknown output '{output_selector}'",
                )
            )
        for metric in metrics:
            if metric not in supported_metrics:
                diagnostics.append(
                    _error(
                        "invalid_compile_metric",
                        f"Metric '{metric}' is not supported by the Keras runtime",
                    )
                )

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
    return diagnostics
