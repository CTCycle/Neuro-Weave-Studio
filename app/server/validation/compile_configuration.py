from __future__ import annotations

import math
from typing import Any

from domain.diagnostics import ValidationDiagnostic
from domain.enums import CodeTarget, DiagnosticSeverity
from domain.graph import ModelGraph

SUPPORTED_OPTIMIZERS = frozenset({"adam", "adamw", "sgd", "rmsprop"})
SUPPORTED_LOSSES = frozenset(
    {
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "binary_crossentropy",
        "mean_squared_error",
        "mean_absolute_error",
    }
)
SUPPORTED_METRICS = frozenset({"accuracy", "precision", "recall", "auc", "mae", "mse"})
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
    code_target: CodeTarget,
) -> list[ValidationDiagnostic]:
    """Validate the compile settings used by the Keras runtime.

    Other source targets retain their existing target-specific source contract;
    Keras is the executable model API and the only target that consumes this
    configuration in the current product boundary.
    """

    if code_target is not CodeTarget.KERAS:
        return []

    configuration = graph.model_configuration
    diagnostics: list[ValidationDiagnostic] = []
    if configuration.optimizer not in SUPPORTED_OPTIMIZERS:
        diagnostics.append(
            _error(
                "invalid_compile_optimizer",
                f"Optimizer '{configuration.optimizer}' is not supported by the Keras runtime",
            )
        )

    unknown_optimizer_parameters = set(configuration.optimizer_parameters) - {"learning_rate"}
    if unknown_optimizer_parameters:
        diagnostics.append(
            _error(
                "unsupported_optimizer_parameter",
                "Only learning_rate is supported in optimizer_parameters",
            )
        )
    learning_rate: Any = configuration.optimizer_parameters.get("learning_rate", 0.001)
    if (
        not isinstance(learning_rate, int | float)
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        diagnostics.append(
            _error(
                "invalid_optimizer_learning_rate",
                "optimizer_parameters.learning_rate must be a finite positive number",
            )
        )

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
