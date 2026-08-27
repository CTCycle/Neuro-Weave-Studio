from __future__ import annotations

import math

from domain.compilation_catalog import get_compilation_catalog
from domain.diagnostics import ValidationDiagnostic
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import ModelGraph

_CATALOG = get_compilation_catalog()
SUPPORTED_OPTIMIZERS = frozenset(item.id for item in _CATALOG.optimizers)
SUPPORTED_LOSSES = frozenset(item.id for item in _CATALOG.losses)
SUPPORTED_METRICS = frozenset(item.id for item in _CATALOG.metrics)
OPTIMIZER_PARAMETERS = {
    item.id: {parameter.id: parameter for parameter in item.parameters}
    for item in _CATALOG.optimizers
}
SUPPORTED_INITIALIZERS = frozenset(
    {
        "uniform",
        "glorot_uniform",
        "glorot_normal",
        "he_uniform",
        "he_normal",
        "zeros",
        "ones",
        "random_normal",
        "random_uniform",
        "orthogonal",
    }
)
SUPPORTED_REGULARIZERS = frozenset({"none", "l1", "l2", "l1_l2"})
SUPPORTED_CONSTRAINTS = frozenset({"none", "max_norm", "min_max_norm", "non_neg", "unit_norm"})

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
) -> list[ValidationDiagnostic]:
    """Validate the selected backend's canonical compilation configuration."""

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
    if configuration.optimizer not in SUPPORTED_OPTIMIZERS:
        diagnostics.append(
            _error(
                "invalid_compile_optimizer",
                f"Optimizer '{configuration.optimizer}' is not supported by the Keras runtime",
            )
        )

    allowed_parameters = OPTIMIZER_PARAMETERS.get(configuration.optimizer, {})
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
        if loss not in SUPPORTED_LOSSES:
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
            if metric not in SUPPORTED_METRICS:
                diagnostics.append(
                    _error(
                        "invalid_compile_metric",
                        f"Metric '{metric}' is not supported by the Keras runtime",
                    )
                )

    if configuration.default_initializer not in SUPPORTED_INITIALIZERS:
        diagnostics.append(
            _error(
                "invalid_default_initializer",
                f"Initializer '{configuration.default_initializer}' is not supported",
            )
        )
    if configuration.default_regularizer not in SUPPORTED_REGULARIZERS:
        diagnostics.append(
            _error(
                "invalid_default_regularizer",
                f"Regularizer '{configuration.default_regularizer}' is not supported",
            )
        )
    if configuration.default_constraint not in SUPPORTED_CONSTRAINTS:
        diagnostics.append(
            _error(
                "invalid_default_constraint",
                f"Constraint '{configuration.default_constraint}' is not supported",
            )
        )
    return diagnostics
