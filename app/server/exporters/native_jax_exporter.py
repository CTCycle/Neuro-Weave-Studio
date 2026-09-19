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
from domain.compilation_plan import CompilationObject
from domain.graph import ModelGraph
from domain.ports import materialized_outputs
from domain.tensor import TensorSpec
from exporters.base import SourceExporter
from validation.output_resolution import output_selectors, resolved_outputs
from validation.parameter_values import condition_matches
from validation.safe_expression import normalize_index_expression, validate_safe_expression

###############################################################################
class NativeJaxSourceExporter(SourceExporter):
    """Generate a native Flax Linen module for the verified subset."""

    # -------------------------------------------------------------------------
    @staticmethod
    def _summary_map(ir: CanonicalModelIR) -> dict[str, TensorSpec]:
        return {
            summary.node_id: next(iter(summary.outputs.values()))
            for summary in ir.shape_summaries
            if summary.outputs
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _static_dimension(spec: TensorSpec | None, index: int) -> int | None:
        if spec is None:
            return None
        shape = cast(list[Any], spec.shape)
        if not -len(shape) <= index < len(shape):
            return None
        value = shape[index]
        return value if isinstance(value, int) else None

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
    def _catalog_option(
        compilation_catalog: CompilationCatalog,
        category: str,
        identifier: str,
    ) -> CompilationOption:
        option = next(
            (item for item in getattr(compilation_catalog, category) if item.id == identifier),
            None,
        )
        if option is None:
            raise ValueError(f"No JAX catalog option registered for {category}.{identifier}")
        implementation = option.implementations.get("jax")
        if implementation is None or not implementation.supported or not implementation.adapter:
            raise ValueError(f"No JAX adapter registered for {category}.{identifier}")
        return option

    # -------------------------------------------------------------------------
    @classmethod
    def _compilation_payload(
        cls,
        graph: ModelGraph,
        catalog: NodeCatalog,
        compilation_catalog: CompilationCatalog,
    ) -> dict[str, Any]:
        configuration = graph.model_configuration.native.jax.compilation
        outputs = [output for output in resolved_outputs(graph, catalog) if output.node is not None]

        def selected(values: dict[str, Any], output: Any, default: Any) -> Any:
            value = values.get("__default__", default)
            for selector in output_selectors(output):
                if selector and selector in values:
                    value = values[selector]
                    break
            return value

        loss_ids: dict[str, str] = {}
        loss_parameters: dict[str, dict[str, Any]] = {}
        metric_ids: dict[str, list[str]] = {}
        metric_parameters: dict[str, list[dict[str, Any]]] = {}
        target_contracts: dict[str, str] = {}
        metric_target_contracts: dict[str, dict[str, str]] = {}
        losses: dict[str, dict[str, Any]] = {}
        metrics: dict[str, list[dict[str, Any]]] = {}
        for output in outputs:
            loss_value = selected(configuration.losses, output, None)
            if isinstance(loss_value, CompilationObject):
                option = cls._catalog_option(compilation_catalog, "losses", loss_value.id)
                loss_ids[output.name] = loss_value.id
                loss_parameters[output.name] = dict(loss_value.parameters)
                losses[output.name] = {
                    "id": loss_value.id,
                    "parameters": dict(loss_value.parameters),
                }
                if option.target_contract is not None:
                    target_contracts[output.name] = option.target_contract
            metric_values = selected(configuration.metrics, output, [])
            if not isinstance(metric_values, list):
                continue
            metric_ids[output.name] = []
            metric_parameters[output.name] = []
            metrics[output.name] = []
            metric_target_contracts[output.name] = {}
            for metric_value in metric_values:
                if not isinstance(metric_value, CompilationObject):
                    continue
                option = cls._catalog_option(compilation_catalog, "metrics", metric_value.id)
                metric_ids[output.name].append(metric_value.id)
                metric_parameters[output.name].append(dict(metric_value.parameters))
                metrics[output.name].append(
                    {"id": metric_value.id, "parameters": dict(metric_value.parameters)}
                )
                metric_target_contracts[output.name][metric_value.id] = (
                    option.target_contract or "continuous_same_shape"
                )
                target_contracts.setdefault(
                    output.name,
                    option.target_contract or "continuous_same_shape",
                )
        return {
            "loss_ids": loss_ids,
            "loss_parameters": loss_parameters,
            "metric_ids": metric_ids,
            "metric_parameters": metric_parameters,
            "target_contracts": target_contracts,
            "metric_target_contracts": metric_target_contracts,
            "losses": losses,
            "metrics": metrics,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _compilation_formula_lines() -> list[str]:
        return [
            "",
            "",
            "def _loss_formula(prediction, target, kind, **parameters):",
            "    expected = jnp.asarray(target, dtype=prediction.dtype)",
            "    if kind == 'mean_squared_error': return jnp.mean(jnp.square(prediction - expected))",
            "    if kind == 'mean_absolute_error': return jnp.mean(jnp.abs(prediction - expected))",
            "    if kind == 'binary_crossentropy':",
            "        expected = expected * (1.0 - parameters.get('label_smoothing', 0.0)) + 0.5 * parameters.get('label_smoothing', 0.0)",
            "        values = jax.nn.softplus(prediction) - expected * prediction if parameters.get('from_logits', False) else -expected * jnp.log(jnp.clip(prediction, 1e-7, 1.0 - 1e-7)) - (1.0 - expected) * jnp.log(jnp.clip(1.0 - prediction, 1e-7, 1.0))",
            "        axis = parameters.get('axis', -1)",
            "        return jnp.mean(jnp.mean(values, axis=axis)) if values.ndim > 1 else jnp.mean(values)",
            "    if kind == 'categorical_crossentropy':",
            "        axis = parameters.get('axis', -1)",
            "        expected = expected * (1.0 - parameters.get('label_smoothing', 0.0)) + parameters.get('label_smoothing', 0.0) / prediction.shape[axis]",
            "        log_values = jax.nn.log_softmax(prediction, axis=axis) if parameters.get('from_logits', False) else jnp.log(jnp.clip(prediction, 1e-7, 1.0))",
            "        return -jnp.mean(jnp.sum(expected * log_values, axis=axis))",
            "    if kind == 'sparse_categorical_crossentropy':",
            "        axis = parameters.get('axis', -1)",
            "        labels = jnp.asarray(target, dtype=jnp.int32)",
            "        log_values = jax.nn.log_softmax(prediction, axis=axis) if parameters.get('from_logits', False) else jnp.log(jnp.clip(prediction, 1e-7, 1.0))",
            "        expected = jax.nn.one_hot(labels, prediction.shape[axis], dtype=prediction.dtype)",
            "        if parameters.get('label_smoothing', 0.0) != 0.0: expected = expected * (1.0 - parameters.get('label_smoothing', 0.0)) + parameters.get('label_smoothing', 0.0) / prediction.shape[axis]",
            "        values = -jnp.sum(expected * log_values, axis=axis)",
            "        if parameters.get('ignore_class') is not None: values = jnp.where(labels == parameters['ignore_class'], 0.0, values)",
            "        return jnp.mean(values)",
            "    if kind == 'huber':",
            "        delta = jnp.asarray(parameters.get('delta', 1.0), dtype=prediction.dtype)",
            "        difference = jnp.abs(prediction - expected)",
            "        quadratic = jnp.minimum(difference, delta)",
            "        return jnp.mean(0.5 * jnp.square(quadratic) + delta * (difference - quadratic))",
            "    if kind == 'kullback_leibler_divergence':",
            "        actual = jnp.clip(expected, 1e-7, 1.0)",
            "        predicted = jnp.clip(prediction, 1e-7, 1.0)",
            "        return jnp.mean(jnp.sum(actual * jnp.log(actual / predicted), axis=-1))",
            "    if kind == 'log_cosh':",
            "        difference = jnp.abs(prediction - expected)",
            "        return jnp.mean(jax.nn.softplus(2.0 * difference) - difference - jnp.log(2.0))",
            "    if kind == 'mean_absolute_percentage_error': return jnp.mean(jnp.abs(prediction - expected) / jnp.maximum(jnp.abs(expected), 1e-7)) * 100.0",
            "    if kind == 'mean_squared_logarithmic_error': return jnp.mean(jnp.square(jnp.log1p(jnp.maximum(prediction, 0.0)) - jnp.log1p(jnp.maximum(expected, 0.0))))",
            "    if kind == 'hinge': return jnp.mean(jnp.maximum(0.0, 1.0 - expected * prediction))",
            "    if kind == 'squared_hinge': return jnp.mean(jnp.square(jnp.maximum(0.0, 1.0 - expected * prediction)))",
            "    if kind == 'categorical_hinge':",
            "        positive = jnp.sum(prediction * expected, axis=-1)",
            "        negative = jnp.max((1.0 - expected) * prediction, axis=-1)",
            "        return jnp.mean(jnp.maximum(0.0, negative - positive + 1.0))",
            "    if kind == 'poisson':",
            "        rate = jnp.maximum(prediction, 1e-7)",
            "        return jnp.mean(rate - expected * jnp.log(rate))",
            "    if kind == 'dice':",
            "        axis = parameters.get('axis')",
            "        intersection = jnp.sum(prediction * expected, axis=axis)",
            "        denominator = jnp.sum(prediction, axis=axis) + jnp.sum(expected, axis=axis)",
            "        return jnp.mean(1.0 - 2.0 * intersection / jnp.maximum(denominator, 1e-7))",
            "    if kind == 'tversky':",
            "        axis = parameters.get('axis')",
            "        true_positive = jnp.sum(prediction * expected, axis=axis)",
            "        false_positive = jnp.sum(prediction * (1.0 - expected), axis=axis)",
            "        false_negative = jnp.sum((1.0 - prediction) * expected, axis=axis)",
            "        denominator = true_positive + parameters.get('alpha', 0.5) * false_positive + parameters.get('beta', 0.5) * false_negative",
            "        return jnp.mean(1.0 - true_positive / jnp.maximum(denominator, 1e-7))",
            "    raise ValueError(f'Unsupported JAX loss adapter: {kind}')",
            "",
            "",
            "def _classification_labels(prediction, target):",
            "    if prediction.shape[-1] == 1:",
            "        return (prediction.reshape(-1) >= 0.5).astype(jnp.int32), (jnp.asarray(target).reshape(-1) >= 0.5).astype(jnp.int32)",
            "    predicted = jnp.argmax(prediction, axis=-1).reshape(-1)",
            "    expected = jnp.argmax(target, axis=-1).reshape(-1) if target.shape == prediction.shape else jnp.asarray(target, dtype=jnp.int32).reshape(-1)",
            "    return predicted, expected",
            "",
            "",
            "def _binary_vectors(prediction, target, threshold=0.5, top_k=None, class_id=None):",
            "    if class_id is not None and prediction.shape[-1] > 1:",
            "        scores = prediction[..., class_id]",
            "        expected = target[..., class_id] if target.shape == prediction.shape else (jnp.asarray(target, dtype=jnp.int32) == class_id).astype(prediction.dtype)",
            "        return (scores.reshape(-1) >= threshold).astype(jnp.int32), (expected.reshape(-1) >= 0.5).astype(jnp.int32)",
            "    if top_k is not None and prediction.shape[-1] > 1:",
            "        limit = min(max(int(top_k), 1), int(prediction.shape[-1]))",
            "        candidates = jnp.argsort(prediction, axis=-1)[..., -limit:]",
            "        predicted = jnp.sum(jax.nn.one_hot(candidates, prediction.shape[-1]), axis=-2) > 0",
            "        expected = target >= 0.5 if target.shape == prediction.shape else jax.nn.one_hot(jnp.asarray(target, dtype=jnp.int32), prediction.shape[-1]) > 0",
            "        return predicted.reshape(-1).astype(jnp.int32), expected.reshape(-1).astype(jnp.int32)",
            "    if prediction.shape[-1] == 1:",
            "        expected = target[..., 0] if target.shape == prediction.shape else target",
            "        return (prediction[..., 0].reshape(-1) >= threshold).astype(jnp.int32), (jnp.asarray(expected).reshape(-1) >= 0.5).astype(jnp.int32)",
            "    return _classification_labels(prediction, target)",
            "",
            "",
            "def _auc_binary(scores, labels, parameters):",
            "    expected = labels >= 0.5",
            "    positives = jnp.sum(expected.astype(jnp.float32))",
            "    negatives = jnp.sum((~expected).astype(jnp.float32))",
            "    explicit = parameters.get('thresholds')",
            "    thresholds = jnp.asarray(explicit, dtype=scores.dtype) if explicit else jnp.linspace(0.0, 1.0, max(int(parameters.get('num_thresholds', 200)), 2))",
            "    x_values = []",
            "    y_values = []",
            "    for threshold in thresholds:",
            "        predicted = scores >= threshold",
            "        true_positive = jnp.sum((predicted & expected).astype(jnp.float32))",
            "        false_positive = jnp.sum((predicted & ~expected).astype(jnp.float32))",
            "        if parameters.get('curve', 'ROC') == 'PR':",
            "            x_values.append(true_positive / jnp.maximum(positives, 1.0))",
            "            y_values.append(true_positive / jnp.maximum(predicted.astype(jnp.float32).sum(), 1.0))",
            "        else:",
            "            x_values.append(false_positive / jnp.maximum(negatives, 1.0))",
            "            y_values.append(true_positive / jnp.maximum(positives, 1.0))",
            "    x = jnp.stack(x_values)",
            "    y = jnp.stack(y_values)",
            "    order = jnp.argsort(x)",
            "    x = x[order]",
            "    y = y[order]",
            "    widths = jnp.abs(x[1:] - x[:-1])",
            "    lower = jnp.minimum(y[1:], y[:-1])",
            "    upper = jnp.maximum(y[1:], y[:-1])",
            "    heights = lower if parameters.get('summation_method', 'interpolation') == 'minoring' else upper if parameters.get('summation_method') == 'majoring' else (y[1:] + y[:-1]) * 0.5",
            "    return jnp.clip(jnp.sum(widths * heights), 0.0, 1.0)",
            "",
            "",
            "def _metric_formula(prediction, target, kind, **parameters):",
            "    expected = jnp.asarray(target, dtype=prediction.dtype)",
            "    if kind == 'accuracy':",
            "        predicted, labels = _classification_labels(prediction, target)",
            "        return jnp.mean((predicted == labels).astype(jnp.float32))",
            "    if kind == 'binary_accuracy': return jnp.mean((prediction >= parameters.get('threshold', 0.5)) == (expected >= 0.5))",
            "    if kind == 'categorical_accuracy': return jnp.mean((jnp.argmax(prediction, -1) == jnp.argmax(expected, -1)).astype(jnp.float32))",
            "    if kind == 'sparse_categorical_accuracy': return jnp.mean((jnp.argmax(prediction, -1) == jnp.asarray(target, dtype=jnp.int32)).astype(jnp.float32))",
            "    if kind in ('top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy'):",
            "        limit = min(max(int(parameters.get('k', 5)), 1), int(prediction.shape[-1]))",
            "        candidates = jnp.argsort(prediction, axis=-1)[..., -limit:]",
            "        labels = jnp.argmax(expected, -1) if kind == 'top_k_categorical_accuracy' else jnp.asarray(target, dtype=jnp.int32)",
            "        return jnp.mean(jnp.any(candidates == labels[..., None], axis=-1).astype(jnp.float32))",
            "    if kind in ('precision', 'recall'):",
            "        predicted, labels = _binary_vectors(prediction, target, parameters.get('threshold', 0.5), parameters.get('top_k'), parameters.get('class_id'))",
            "        true_positive = jnp.sum(((predicted == 1) & (labels == 1)).astype(jnp.float32))",
            "        denominator = jnp.sum((predicted == 1).astype(jnp.float32)) if kind == 'precision' else jnp.sum((labels == 1).astype(jnp.float32))",
            "        return true_positive / jnp.maximum(denominator, 1.0)",
            "    if kind == 'auc':",
            "        scores = jax.nn.sigmoid(prediction) if parameters.get('from_logits', False) else prediction",
            "        if parameters.get('multi_label', False):",
            "            scores = scores.reshape(-1, scores.shape[-1])",
            "            labels = expected.reshape(-1, expected.shape[-1])",
            "            values = jnp.stack([_auc_binary(scores[:, index], labels[:, index], parameters) for index in range(int(scores.shape[-1]))])",
            "            weights = parameters.get('label_weights')",
            "            return jnp.sum(values * jnp.asarray(weights)) / jnp.maximum(jnp.sum(jnp.asarray(weights)), 1e-7) if weights else jnp.mean(values)",
            "        scores = scores[..., 0] if scores.shape[-1] == 1 else scores[..., 1]",
            "        labels = expected[..., 0] if expected.shape[-1] == 1 else expected[..., 1] if target.shape == prediction.shape else expected",
            "        return _auc_binary(scores.reshape(-1), labels.reshape(-1), parameters)",
            "    if kind in ('true_positives', 'true_negatives', 'false_positives', 'false_negatives'):",
            "        predicted, labels = _binary_vectors(prediction, target, parameters.get('threshold', 0.5))",
            "        if kind == 'true_positives': return jnp.sum(((predicted == 1) & (labels == 1)).astype(jnp.float32))",
            "        if kind == 'true_negatives': return jnp.sum(((predicted == 0) & (labels == 0)).astype(jnp.float32))",
            "        if kind == 'false_positives': return jnp.sum(((predicted == 1) & (labels == 0)).astype(jnp.float32))",
            "        return jnp.sum(((predicted == 0) & (labels == 1)).astype(jnp.float32))",
            "    if kind == 'mean_absolute_error': return jnp.mean(jnp.abs(prediction - expected))",
            "    if kind == 'mean_squared_error': return jnp.mean(jnp.square(prediction - expected))",
            "    if kind == 'root_mean_squared_error': return jnp.sqrt(jnp.mean(jnp.square(prediction - expected)))",
            "    if kind == 'mean_absolute_percentage_error': return jnp.mean(jnp.abs(prediction - expected) / jnp.maximum(jnp.abs(expected), 1e-7)) * 100.0",
            "    if kind == 'mean_squared_logarithmic_error': return jnp.mean(jnp.square(jnp.log1p(jnp.maximum(prediction, 0.0)) - jnp.log1p(jnp.maximum(expected, 0.0))))",
            "    if kind == 'cosine_similarity':",
            "        axis = parameters.get('axis', -1)",
            "        return jnp.mean(jnp.sum((prediction / jnp.maximum(jnp.linalg.norm(prediction, axis=axis, keepdims=True), 1e-7)) * (expected / jnp.maximum(jnp.linalg.norm(expected, axis=axis, keepdims=True), 1e-7)), axis=axis))",
            "    if kind == 'log_cosh_error':",
            "        difference = jnp.abs(prediction - expected)",
            "        return jnp.mean(jax.nn.softplus(2.0 * difference) - difference - jnp.log(2.0))",
            "    if kind == 'prediction_mean': return jnp.mean(prediction)",
            "    if kind == 'prediction_sum': return jnp.sum(prediction)",
            "    raise ValueError(f'Unsupported JAX metric adapter: {kind}')",
            "",
            "",
            "_LOSS_ADAPTERS = {",
            "    identifier: functools.partial(_loss_formula, kind=identifier) for identifier in [",
            "        'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'categorical_crossentropy',",
            "        'sparse_categorical_crossentropy', 'huber', 'kullback_leibler_divergence', 'log_cosh',",
            "        'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'hinge', 'squared_hinge',",
            "        'categorical_hinge', 'poisson', 'dice', 'tversky',",
            "    ]",
            "}",
            "_METRIC_ADAPTERS = {",
            "    identifier: functools.partial(_metric_formula, kind=identifier) for identifier in [",
            "        'accuracy', 'binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy',",
            "        'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy', 'precision', 'recall', 'auc',",
            "        'true_positives', 'true_negatives', 'false_positives', 'false_negatives',",
            "        'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error',",
            "        'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'cosine_similarity',",
            "        'log_cosh_error', 'prediction_mean', 'prediction_sum',",
            "    ]",
            "}",
        ]

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
    @staticmethod
    def _spatial_values(
        parameters: dict[str, Any],
        key: str,
        dimensions: int,
        default: list[int],
    ) -> tuple[int, ...]:
        value = parameters.get(key, default)
        if value is None:
            value = default
        if not isinstance(value, list) or len(value) != dimensions:
            raise ValueError(f"JAX {key} expects {dimensions} values")
        values = cast(list[Any], value)
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in values):
            raise ValueError(f"JAX {key} expects positive integer values")
        return tuple(int(item) for item in values)

    # -------------------------------------------------------------------------
    @staticmethod
    def _spatial_nonnegative(
        parameters: dict[str, Any],
        key: str,
        dimensions: int,
        default: list[int],
    ) -> tuple[int, ...]:
        value = parameters.get(key, default)
        if value is None:
            value = default
        if not isinstance(value, list) or len(value) != dimensions:
            raise ValueError(f"JAX {key} expects {dimensions} values")
        values = cast(list[Any], value)
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
            raise ValueError(f"JAX {key} expects non-negative integer values")
        return tuple(int(item) for item in values)

    # -------------------------------------------------------------------------
    @staticmethod
    def _spatial_pairs(parameters: dict[str, Any], key: str, dimensions: int) -> tuple[tuple[int, int], ...]:
        value = parameters.get(key)
        if not isinstance(value, list) or len(value) != dimensions:
            raise ValueError(f"JAX {key} expects {dimensions} pairs")
        pairs: list[tuple[int, int]] = []
        for item in cast(list[Any], value):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(not isinstance(entry, int) or isinstance(entry, bool) or entry < 0 for entry in item)
            ):
                raise ValueError(f"JAX {key} expects non-negative pairs")
            pairs.append((int(item[0]), int(item[1])))
        return tuple(pairs)

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
        summaries = self._summary_map(ir)
        lines = [
            "from __future__ import annotations",
            "",
            "import functools",
            "import jax",
            "import jax.numpy as jnp",
            "import optax",
            "from flax import linen as nn",
            "",
            "",
            "def _pool_nd(value, pool_size, strides, padding, data_format, mode):",
            "    dimensions = len(pool_size)",
            "    channels_first = data_format == 'channels_first'",
            "    if channels_first:",
            "        value = jnp.transpose(value, (0, *range(2, dimensions + 2), 1))",
            "    operation = nn.max_pool if mode == 'max' else nn.avg_pool",
            "    current = operation(value, window_shape=pool_size, strides=strides, padding=padding.upper())",
            "    if channels_first:",
            "        current = jnp.transpose(current, (0, dimensions + 1, *range(1, dimensions + 1)))",
            "    return current",
            "",
            "",
            "def _global_pool_nd(value, dimensions, data_format, mode, keepdims):",
            "    channels_first = data_format == 'channels_first'",
            "    if channels_first:",
            "        value = jnp.transpose(value, (0, *range(2, dimensions + 2), 1))",
            "    axes = tuple(range(1, dimensions + 1))",
            "    current = jnp.mean(value, axis=axes, keepdims=keepdims) if mode == 'average' else jnp.max(value, axis=axes, keepdims=keepdims)",
            "    if channels_first and keepdims:",
            "        current = jnp.transpose(current, (0, dimensions + 1, *range(1, dimensions + 1)))",
            "    return current",
            "",
            "",
            "def _sinusoidal_encoding(value):",
            "    positions = jnp.arange(value.shape[1], dtype=value.dtype)[:, None]",
            "    indices = jnp.arange(value.shape[2], dtype=value.dtype)[None, :]",
            "    dimension = jnp.asarray(value.shape[2], dtype=value.dtype)",
            "    rates = jnp.power(jnp.asarray(10000.0, dtype=value.dtype), (2.0 * jnp.floor(indices / 2.0)) / dimension)",
            "    angles = positions / rates",
            "    even = jnp.equal(jnp.mod(indices.astype(jnp.int32), 2), 0)",
            "    encoding = jnp.where(even, jnp.sin(angles), jnp.cos(angles))",
            "    return value + encoding[None, :, :]",
            "",
            "",
            "def _scaled_dot_product_attention(query, key, value, heads, key_dim, value_dim):",
            "    query = query.reshape(query.shape[0], query.shape[1], heads, key_dim).transpose(0, 2, 1, 3)",
            "    key = key.reshape(key.shape[0], key.shape[1], heads, key_dim).transpose(0, 2, 1, 3)",
            "    value = value.reshape(value.shape[0], value.shape[1], heads, value_dim).transpose(0, 2, 1, 3)",
            "    scores = jnp.einsum('bhqd,bhkd->bhqk', query, key) / (key_dim ** 0.5)",
            "    weights = jax.nn.softmax(scores, axis=-1)",
            "    attended = jnp.einsum('bhqk,bhkv->bhqv', weights, value)",
            "    return attended.transpose(0, 2, 1, 3).reshape(query.shape[0], -1, heads * value_dim)",
            "",
            "",
            "def _upsample_nd(value, size, data_format, interpolation):",
            "    dimensions = len(size)",
            "    channels_first = data_format == 'channels_first'",
            "    if channels_first:",
            "        value = jnp.transpose(value, (0, *range(2, dimensions + 2), 1))",
            "    if interpolation == 'bilinear' and dimensions == 2:",
            "        target_shape = (value.shape[0], value.shape[1] * size[0], value.shape[2] * size[1], value.shape[3])",
            "        current = jax.image.resize(value, target_shape, method='linear')",
            "    else:",
            "        current = value",
            "        for axis, multiplier in enumerate(size, start=1):",
            "            current = jnp.repeat(current, multiplier, axis=axis)",
            "    if channels_first:",
            "        current = jnp.transpose(current, (0, dimensions + 1, *range(1, dimensions + 1)))",
            "    return current",
            "",
            "",
            "def _pad_nd(value, pairs, data_format):",
            "    dimensions = len(pairs)",
            "    if data_format == 'channels_first':",
            "        pad_width = [(0, 0), (0, 0), *pairs]",
            "    else:",
            "        pad_width = [(0, 0), *pairs, (0, 0)]",
            "    return jnp.pad(value, pad_width)",
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
            source_by_port = {
                binding.target_port_id: port_variables[
                    (binding.source_node_id, binding.source_port_id)
                ]
                for binding in item.input_bindings
            }
            source_spec = summaries.get(item.input_node_ids[0]) if item.input_node_ids else None
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
            elif definition in {"conv.conv3d", "conv.depthwise", "conv.separable"}:
                dimensions = 3 if definition == "conv.conv3d" else int(params.get("dimensions", 2))
                if source_spec is None or source_spec.rank != dimensions + 2:
                    raise ValueError(f"Native JAX {definition} node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                channel_index = dimensions + 1 if data_format == "channels_last" else 1
                input_channels = self._static_dimension(source_spec, channel_index)
                if input_channels is None:
                    raise ValueError(f"Native JAX {definition} node {node.id} requires known input channels")
                if data_format == "channels_first":
                    lines.append(f"        {name}_input = {source}.transpose(0, *range(2, {dimensions + 2}), 1)")
                else:
                    lines.append(f"        {name}_input = {source}")
                padding = str(params.get("padding", "same")).upper()
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                use_bias = bool(params.get("use_bias", True))
                if definition == "conv.conv3d":
                    lines.append(
                        f"        {name} = nn.Conv(features={int(params.get('filters', 32))}, kernel_size={kernel_size!r}, strides={strides!r}, kernel_dilation={dilation!r}, padding={padding!r}, feature_group_count={int(params.get('groups', 1))}, use_bias={use_bias!r}, name={name!r})({name}_input)"
                    )
                elif definition == "conv.depthwise":
                    depth_multiplier = int(params.get("depth_multiplier", 1))
                    lines.append(
                        f"        {name} = nn.Conv(features={input_channels * depth_multiplier}, kernel_size={kernel_size!r}, strides={strides!r}, kernel_dilation={dilation!r}, padding={padding!r}, feature_group_count={input_channels}, use_bias={use_bias!r}, name={name!r})({name}_input)"
                    )
                else:
                    depth_multiplier = int(params.get("depth_multiplier", 1))
                    depth_channels = input_channels * depth_multiplier
                    lines.append(
                        f"        {name}_depthwise = nn.Conv(features={depth_channels}, kernel_size={kernel_size!r}, strides={strides!r}, kernel_dilation={dilation!r}, padding={padding!r}, feature_group_count={input_channels}, use_bias=False, name={name + '_depthwise'!r})({name}_input)"
                    )
                    lines.append(
                        f"        {name} = nn.Conv(features={int(params.get('filters', 32))}, kernel_size={((1,) * dimensions)!r}, strides={((1,) * dimensions)!r}, padding='VALID', use_bias={use_bias!r}, name={name!r})({name}_depthwise)"
                    )
                if data_format == "channels_first":
                    lines.append(f"        {name} = {name}.transpose(0, {dimensions + 1}, *range(1, {dimensions + 1}))")
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
            elif definition == "conv.transposed":
                dimensions = int(params.get("dimensions", 2))
                if source_spec is None or source_spec.rank != dimensions + 2:
                    raise ValueError(f"Native JAX transposed convolution node {node.id} expects a rank {dimensions + 2} input")
                data_format = str(params.get("data_format", "channels_last"))
                if data_format == "channels_first":
                    lines.append(f"        {name}_input = {source}.transpose(0, *range(2, {dimensions + 2}), 1)")
                else:
                    lines.append(f"        {name}_input = {source}")
                kernel_size = self._spatial_values(params, "kernel_size", dimensions, [3] * dimensions)
                strides = self._spatial_values(params, "strides", dimensions, [1] * dimensions)
                dilation = self._spatial_values(params, "dilation_rate", dimensions, [1] * dimensions)
                output_padding = self._spatial_nonnegative(params, "output_padding", dimensions, [0] * dimensions)
                padding = str(params.get("padding", "valid")).upper()
                lines.append(
                    f"        {name} = nn.ConvTranspose(features={int(params.get('filters', 32))}, kernel_size={kernel_size!r}, strides={strides!r}, padding={padding!r}, kernel_dilation={dilation!r}, use_bias={bool(params.get('use_bias', True))!r}, name={name!r})({name}_input)"
                )
                if any(output_padding):
                    pad_width = [(0, 0), *[(0, value) for value in output_padding], (0, 0)]
                    lines.append(f"        {name} = jnp.pad({name}, {pad_width!r})")
                if data_format == "channels_first":
                    lines.append(f"        {name} = {name}.transpose(0, {dimensions + 1}, *range(1, {dimensions + 1}))")
                activation = str(params.get("activation", "linear"))
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
            elif definition in {
                "conv.max_pool1d",
                "conv.max_pool2d",
                "conv.max_pool3d",
                "conv.average_pool1d",
                "conv.average_pool2d",
                "conv.average_pool3d",
            }:
                dimensions = int(definition[-2])
                pool_size = self._spatial_values(params, "pool_size", dimensions, [2] * dimensions)
                strides_value = params.get("strides")
                strides = (
                    pool_size
                    if strides_value is None
                    else self._spatial_values(params, "strides", dimensions, list(pool_size))
                )
                padding = str(params.get("padding", "valid"))
                data_format = str(params.get("data_format", "channels_last"))
                mode = "max" if definition.startswith("conv.max") else "avg"
                lines.append(
                    f"        {name} = _pool_nd({source}, {pool_size!r}, {strides!r}, {padding!r}, {data_format!r}, {mode!r})"
                )
            elif definition in {
                "conv.global_pool1d",
                "conv.global_pool2d",
                "conv.global_pool3d",
            }:
                dimensions = int(definition[-2])
                mode = str(params.get("mode", "average"))
                data_format = str(params.get("data_format", "channels_last"))
                keepdims = bool(params.get("keepdims", False))
                lines.append(
                    f"        {name} = _global_pool_nd({source}, {dimensions!r}, {data_format!r}, {mode!r}, {keepdims!r})"
                )
            elif definition == "conv.upsampling":
                dimensions = int(params.get("dimensions", 2))
                size = self._spatial_values(params, "size", dimensions, [2] * dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                interpolation = str(params.get("interpolation", "nearest"))
                lines.append(
                    f"        {name} = _upsample_nd({source}, {size!r}, {data_format!r}, {interpolation!r})"
                )
            elif definition == "conv.padding":
                dimensions = int(params.get("dimensions", 2))
                pairs = self._spatial_pairs(params, "padding", dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                lines.append(f"        {name} = _pad_nd({source}, {pairs!r}, {data_format!r})")
            elif definition == "conv.cropping":
                dimensions = int(params.get("dimensions", 2))
                pairs = self._spatial_pairs(params, "cropping", dimensions)
                data_format = str(params.get("data_format", "channels_last"))
                rank = dimensions + 2
                spatial_indices = (
                    list(range(1, dimensions + 1))
                    if data_format == "channels_last"
                    else list(range(2, dimensions + 2))
                )
                slices = ["slice(None)"] * rank
                for index, (before, after) in zip(spatial_indices, pairs, strict=True):
                    slices[index] = f"slice({before}, {-after if after else 'None'})"
                lines.append(f"        {name} = {source}[({', '.join(slices)},)]")
            elif definition in {
                "recurrent.simple_rnn",
                "recurrent.bidirectional",
                "recurrent.stack",
            }:
                if (
                    params.get("activation", "tanh") != "tanh"
                    or params.get("recurrent_activation", "sigmoid") != "sigmoid"
                ):
                    raise ValueError("Native JAX recurrent export supports tanh/sigmoid only")
                if (
                    float(params.get("dropout", 0.0)) != 0.0
                    or float(params.get("recurrent_dropout", 0.0)) != 0.0
                ):
                    raise ValueError("Native JAX recurrent export does not support recurrent dropout")
                if (
                    params.get("go_backwards", False)
                    or params.get("stateful", False)
                    or params.get("unroll", False)
                ):
                    raise ValueError(
                        "Native JAX recurrent export does not support go_backwards, stateful, or unroll"
                    )
                units = int(params.get("units", 64))
                cell = (
                    "simple_rnn"
                    if definition == "recurrent.simple_rnn"
                    else str(params.get("cell", "lstm"))
                )
                cell_class = {
                    "simple_rnn": "SimpleCell",
                    "lstm": "LSTMCell",
                    "gru": "GRUCell",
                }.get(cell)
                if cell_class is None:
                    raise ValueError(f"Unsupported recurrent cell: {cell}")
                if definition == "recurrent.simple_rnn":
                    return_sequences = bool(params.get("return_sequences", False))
                    if params.get("return_state", False):
                        lines.append(
                            f"        {name}_carry, {name}_sequence = nn.RNN(nn.SimpleCell(features={units}), return_carry=True, name={name!r})({source})"
                        )
                        if return_sequences:
                            lines.append(f"        {name} = ({name}_sequence, {name}_carry)")
                        else:
                            lines.append(f"        {name} = ({name}_carry, {name}_carry)")
                    else:
                        lines.append(
                            f"        {name} = nn.RNN(nn.SimpleCell(features={units}), return_carry=False, name={name!r})({source})"
                        )
                        if not return_sequences:
                            lines.append(f"        {name} = {name}[:, -1, :]")
                elif definition == "recurrent.bidirectional":
                    merge_mode = str(params.get("merge_mode", "concat"))
                    if merge_mode not in {"sum", "mul", "concat", "ave"}:
                        raise ValueError(f"Unsupported bidirectional merge mode: {merge_mode}")
                    if params.get("return_state", False):
                        raise ValueError("Native JAX bidirectional export does not expose recurrent state ports")
                    lines.append(
                        f"        {name}_forward = nn.RNN(nn.{cell_class}(features={units}), return_carry=False, name={name + '_forward'!r})({source})"
                    )
                    lines.append(
                        f"        {name}_backward = nn.RNN(nn.{cell_class}(features={units}), return_carry=False, reverse=True, keep_order=True, name={name + '_backward'!r})({source})"
                    )
                    if merge_mode == "sum":
                        lines.append(f"        {name} = {name}_forward + {name}_backward")
                    elif merge_mode == "mul":
                        lines.append(f"        {name} = {name}_forward * {name}_backward")
                    elif merge_mode == "ave":
                        lines.append(
                            f"        {name} = ({name}_forward + {name}_backward) / 2.0"
                        )
                    else:
                        lines.append(
                            f"        {name} = jnp.concatenate([{name}_forward, {name}_backward], axis=-1)"
                        )
                    if not bool(params.get("return_sequences", False)):
                        lines.append(f"        {name} = {name}[:, -1, :]")
                else:
                    if str(params.get("merge_mode", "concat")) != "concat":
                        raise ValueError("Recurrent stack merge_mode must remain concat")
                    if params.get("return_state", False):
                        raise ValueError("Native JAX recurrent stack does not expose recurrent state ports")
                    layers = int(params.get("layers", 2))
                    if layers < 1:
                        raise ValueError("Recurrent stack layers must be positive")
                    current = source
                    for layer_index in range(layers):
                        layer_name = f"{name}_layer{layer_index}"
                        lines.append(
                            f"        {layer_name} = nn.RNN(nn.{cell_class}(features={units}), return_carry=False, name={layer_name!r})({current})"
                        )
                        current = layer_name
                    lines.append(f"        {name} = {current}")
                    if not bool(params.get("return_sequences", False)):
                        lines.append(f"        {name} = {name}[:, -1, :]")
            elif definition in {"attention.multi_head", "attention.self", "attention.cross"}:
                if int(params.get("layers", 1)) != 1:
                    raise ValueError("Native JAX attention supports layers=1 only")
                query = source_by_port.get("query", source)
                context = source_by_port.get("context", query)
                query_spec = source_spec
                context_spec = source_spec
                for binding in item.input_bindings:
                    binding_spec = summaries.get(binding.source_node_id)
                    if binding.target_port_id == "query":
                        query_spec = binding_spec
                    elif binding.target_port_id == "context":
                        context_spec = binding_spec
                if (
                    query_spec is None
                    or context_spec is None
                    or query_spec.rank != 3
                    or context_spec.rank != 3
                ):
                    raise ValueError(f"Native JAX attention node {node.id} requires rank-3 query and context inputs")
                query_dimension = self._static_dimension(query_spec, -1)
                if query_dimension is None:
                    raise ValueError(f"Native JAX attention node {node.id} requires a known query embedding dimension")
                raw_output_shape = params.get("output_shape")
                if raw_output_shape is None:
                    output_shape = None
                    output_dimension = query_dimension
                elif isinstance(raw_output_shape, int) and not isinstance(raw_output_shape, bool):
                    output_shape = (int(raw_output_shape),)
                    output_dimension = int(raw_output_shape)
                elif isinstance(raw_output_shape, (list, tuple)) and raw_output_shape:
                    output_shape = tuple(int(value) for value in raw_output_shape)
                    output_dimension = 1
                    for value in output_shape:
                        output_dimension *= value
                else:
                    raise ValueError(f"Native JAX attention node {node.id} has an invalid output_shape")
                heads = int(params.get("num_heads", 8))
                key_dimension = int(params.get("key_dim", 64))
                value_dimension = int(params.get("value_dim", key_dimension) or key_dimension)
                use_bias = bool(params.get("use_bias", True))
                lines.append(
                    f"        {name}_query = nn.Dense(features={heads * key_dimension}, use_bias={use_bias!r}, name={name + '_query'!r})({query})"
                )
                lines.append(
                    f"        {name}_key = nn.Dense(features={heads * key_dimension}, use_bias={use_bias!r}, name={name + '_key'!r})({context})"
                )
                lines.append(
                    f"        {name}_value = nn.Dense(features={heads * value_dimension}, use_bias={use_bias!r}, name={name + '_value'!r})({context})"
                )
                lines.append(
                    f"        {name}_attended = _scaled_dot_product_attention({name}_query, {name}_key, {name}_value, {heads}, {key_dimension}, {value_dimension})"
                )
                lines.append(
                    f"        {name} = nn.Dense(features={output_dimension}, use_bias={use_bias!r}, name={name + '_output'!r})({name}_attended)"
                )
                if output_shape is not None:
                    lines.append(
                        f"        {name} = {name}.reshape(({name}.shape[0], {name}.shape[1]) + {output_shape!r})"
                    )
            elif definition == "transformer.feed_forward":
                if source_spec is None or source_spec.rank != 3:
                    raise ValueError(f"Native JAX feed-forward node {node.id} requires a rank-3 input")
                if float(params.get("dropout", 0.0)) != 0.0:
                    raise ValueError("Native JAX feed-forward export supports dropout=0 only")
                input_dimension = self._static_dimension(source_spec, -1)
                if input_dimension is None:
                    raise ValueError(f"Native JAX feed-forward node {node.id} requires a known embedding dimension")
                hidden_dimension = input_dimension * int(params.get("ffn_multiplier", 4))
                activation = str(params.get("activation", "gelu"))
                lines.append(
                    f"        {name} = nn.Dense(features={hidden_dimension}, name={name + '_expand'!r})({source})"
                )
                if activation != "linear":
                    lines.append(f"        {name} = {self._activation(activation)}({name})")
                lines.append(
                    f"        {name} = nn.Dense(features={input_dimension}, name={name + '_project'!r})({name})"
                )
            elif definition in {"transformer.encoder_block", "transformer.decoder_block", "transformer.stack"}:
                query = source_by_port.get("query", source)
                context = source_by_port.get("context", query)
                query_spec = source_spec
                context_spec = source_spec
                for binding in item.input_bindings:
                    binding_spec = summaries.get(binding.source_node_id)
                    if binding.target_port_id == "query":
                        query_spec = binding_spec
                    elif binding.target_port_id == "context":
                        context_spec = binding_spec
                if query_spec is None or query_spec.rank != 3:
                    raise ValueError(f"Native JAX {definition} node {node.id} requires a rank-3 query input")
                if definition == "transformer.decoder_block" and (context_spec is None or context_spec.rank != 3):
                    raise ValueError(f"Native JAX {definition} node {node.id} requires a rank-3 context input")
                query_dimension = self._static_dimension(query_spec, -1)
                if query_dimension is None:
                    raise ValueError(f"Native JAX {definition} node {node.id} requires a known query dimension")
                layer_count = int(params.get("layers", 1))
                if layer_count < 1:
                    raise ValueError("Transformer block layers must be positive")
                heads = int(params.get("num_heads", 8))
                key_dimension = int(params.get("key_dim", 64))
                hidden_dimension = query_dimension * int(params.get("ffn_multiplier", 4))
                activation = str(params.get("activation", "gelu"))
                pre_norm = bool(params.get("pre_norm", True))
                if float(params.get("dropout", 0.0)) != 0.0:
                    raise ValueError("Native JAX transformer blocks support dropout=0 only")
                current = query
                for layer_index in range(layer_count):
                    prefix = f"{name}_layer{layer_index}"
                    if definition in {"transformer.encoder_block", "transformer.stack"}:
                        attention_input = f"{prefix}_attention_input"
                        if pre_norm:
                            lines.append(
                                f"        {attention_input} = nn.LayerNorm(epsilon=0.001, name={prefix + '_attention_norm'!r})({current})"
                            )
                        else:
                            lines.append(f"        {attention_input} = {current}")
                        lines.append(
                            f"        {prefix}_self_query = nn.Dense(features={heads * key_dimension}, name={prefix + '_self_query'!r})({attention_input})"
                        )
                        lines.append(
                            f"        {prefix}_self_key = nn.Dense(features={heads * key_dimension}, name={prefix + '_self_key'!r})({attention_input})"
                        )
                        lines.append(
                            f"        {prefix}_self_value = nn.Dense(features={heads * key_dimension}, name={prefix + '_self_value'!r})({attention_input})"
                        )
                        lines.append(
                            f"        {prefix}_self_attended = _scaled_dot_product_attention({prefix}_self_query, {prefix}_self_key, {prefix}_self_value, {heads}, {key_dimension}, {key_dimension})"
                        )
                        lines.append(
                            f"        {prefix}_attention = nn.Dense(features={query_dimension}, name={prefix + '_self_output'!r})({prefix}_self_attended)"
                        )
                        attention_residual = f"{prefix}_attention_residual"
                        lines.append(f"        {attention_residual} = {current} + {prefix}_attention")
                        norm_name = "ffn_norm" if pre_norm else "attention_norm"
                        ffn_input = f"{prefix}_ffn_input"
                        lines.append(
                            f"        {ffn_input} = nn.LayerNorm(epsilon=0.001, name={prefix + '_' + norm_name!r})({attention_residual})"
                        )
                        current_cross = attention_residual
                    else:
                        self_input = f"{prefix}_self_input"
                        if pre_norm:
                            lines.append(
                                f"        {self_input} = nn.LayerNorm(epsilon=0.001, name={prefix + '_self_norm'!r})({current})"
                            )
                        else:
                            lines.append(f"        {self_input} = {current}")
                        lines.append(
                            f"        {prefix}_self_query = nn.Dense(features={heads * key_dimension}, name={prefix + '_self_query'!r})({self_input})"
                        )
                        lines.append(
                            f"        {prefix}_self_key = nn.Dense(features={heads * key_dimension}, name={prefix + '_self_key'!r})({self_input})"
                        )
                        lines.append(
                            f"        {prefix}_self_value = nn.Dense(features={heads * key_dimension}, name={prefix + '_self_value'!r})({self_input})"
                        )
                        lines.append(
                            f"        {prefix}_self_attended = _scaled_dot_product_attention({prefix}_self_query, {prefix}_self_key, {prefix}_self_value, {heads}, {key_dimension}, {key_dimension})"
                        )
                        lines.append(
                            f"        {prefix}_self_attention = nn.Dense(features={query_dimension}, name={prefix + '_self_output'!r})({prefix}_self_attended)"
                        )
                        self_residual = f"{prefix}_self_residual"
                        lines.append(f"        {self_residual} = {current} + {prefix}_self_attention")
                        current_self = f"{prefix}_current_self"
                        if pre_norm:
                            lines.append(f"        {current_self} = {self_residual}")
                        else:
                            lines.append(
                                f"        {current_self} = nn.LayerNorm(epsilon=0.001, name={prefix + '_self_norm'!r})({self_residual})"
                            )
                        cross_input = f"{prefix}_cross_input"
                        if pre_norm:
                            lines.append(
                                f"        {cross_input} = nn.LayerNorm(epsilon=0.001, name={prefix + '_cross_norm'!r})({current_self})"
                            )
                        else:
                            lines.append(f"        {cross_input} = {current_self}")
                        lines.append(
                            f"        {prefix}_cross_query = nn.Dense(features={heads * key_dimension}, name={prefix + '_cross_query'!r})({cross_input})"
                        )
                        lines.append(
                            f"        {prefix}_cross_key = nn.Dense(features={heads * key_dimension}, name={prefix + '_cross_key'!r})({context})"
                        )
                        lines.append(
                            f"        {prefix}_cross_value = nn.Dense(features={heads * key_dimension}, name={prefix + '_cross_value'!r})({context})"
                        )
                        lines.append(
                            f"        {prefix}_cross_attended = _scaled_dot_product_attention({prefix}_cross_query, {prefix}_cross_key, {prefix}_cross_value, {heads}, {key_dimension}, {key_dimension})"
                        )
                        lines.append(
                            f"        {prefix}_cross_attention = nn.Dense(features={query_dimension}, name={prefix + '_cross_output'!r})({prefix}_cross_attended)"
                        )
                        cross_residual = f"{prefix}_cross_residual"
                        lines.append(f"        {cross_residual} = {current_self} + {prefix}_cross_attention")
                        current_cross = f"{prefix}_current_cross"
                        if pre_norm:
                            lines.append(f"        {current_cross} = {cross_residual}")
                        else:
                            lines.append(
                                f"        {current_cross} = nn.LayerNorm(epsilon=0.001, name={prefix + '_cross_norm'!r})({cross_residual})"
                            )
                        ffn_input = f"{prefix}_ffn_input"
                        if pre_norm:
                            lines.append(
                                f"        {ffn_input} = nn.LayerNorm(epsilon=0.001, name={prefix + '_ffn_norm'!r})({current_cross})"
                            )
                        else:
                            lines.append(f"        {ffn_input} = {current_cross}")
                    lines.append(
                        f"        {prefix}_feed_forward = nn.Dense(features={hidden_dimension}, name={prefix + '_expand'!r})({ffn_input})"
                    )
                    if activation != "linear":
                        lines.append(f"        {prefix}_feed_forward = {self._activation(activation)}({prefix}_feed_forward)")
                    lines.append(
                        f"        {prefix}_feed_forward = nn.Dense(features={query_dimension}, name={prefix + '_project'!r})({prefix}_feed_forward)"
                    )
                    current = f"{prefix}_output"
                    lines.append(f"        {current} = {current_cross} + {prefix}_feed_forward")
                    if not pre_norm:
                        lines.append(
                            f"        {current} = nn.LayerNorm(epsilon=0.001, name={prefix + '_output_norm'!r})({current})"
                        )
                lines.append(f"        {name} = {current}")
            elif definition == "transformer.attention_mask":
                lines.append(f"        {name} = {source}")
            elif definition == "transformer.residual":
                lines.append(f"        {name} = sum([{', '.join(source_vars)}])")
            elif definition == "transformer.positional_encoding":
                if str(params.get("mode", "sinusoidal")) == "learned":
                    if source_spec is None or source_spec.rank != 3:
                        raise ValueError(f"Native JAX positional encoding node {node.id} requires a rank-3 input")
                    sequence_length = self._static_dimension(source_spec, 1)
                    feature_dimension = self._static_dimension(source_spec, 2)
                    if sequence_length is None or feature_dimension is None:
                        raise ValueError(
                            f"Native JAX learned positional encoding node {node.id} requires static sequence and feature dimensions"
                        )
                    embedding = f"{name}_embedding"
                    lines.append(
                        f"        {embedding} = self.param({embedding!r}, nn.initializers.zeros, ({sequence_length}, {feature_dimension}))"
                    )
                    lines.append(f"        {name} = {source} + {embedding}[None, :, :]")
                else:
                    lines.append(f"        {name} = _sinusoidal_encoding({source})")
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
        lines.extend(self._compilation_formula_lines())
        payload = self._compilation_payload(graph, catalog, compilation_catalog)
        jit_enabled = bool(
            graph.model_configuration.native.jax.compiler.parameters.get("enabled", False)
        )
        native = graph.model_configuration.native.jax
        optimizer_value = native.compilation.optimizer
        optimizer_option = next(
            (option for option in compilation_catalog.optimizers if option.id == optimizer_value.id),
            None,
        )
        if optimizer_option is None:
            raise ValueError(f"No JAX optimizer catalog option registered for {optimizer_value.id}")
        optimizer_implementation = optimizer_option.implementations.get("jax")
        if optimizer_implementation is None or not optimizer_implementation.supported:
            raise ValueError(f"No JAX optimizer implementation registered for {optimizer_value.id}")
        accepted_parameters = {parameter.id for parameter in optimizer_option.parameters}
        optimizer_parameters = {
            key: value
            for key, value in optimizer_value.parameters.items()
            if key in accepted_parameters
        }
        weight_decay_supported = "weight_decay" in optimizer_implementation.parameter_mapping
        weight_decay_value = optimizer_parameters.get("weight_decay")
        optimizer_arguments = ", ".join(
            f"{optimizer_implementation.parameter_mapping.get(key, key)}={value!r}"
            for key, value in optimizer_parameters.items()
            if key != "weight_decay" or weight_decay_supported
        )
        lines.extend(
            [
                "",
                "",
                "def build_optimizer():",
                f"    optimizer_runtime = optax.{optimizer_implementation.adapter}({optimizer_arguments})",
                *(
                    [
                        f"    optimizer_runtime = optax.chain(optax.add_decayed_weights({weight_decay_value!r}), optimizer_runtime)"
                    ]
                    if weight_decay_value is not None and not weight_decay_supported
                    else []
                ),
                "    return optimizer_runtime",
                "",
                "",
                "def configured_compilation() -> dict[str, object]:",
                "    return {",
                f"        'optimizer': {{'id': {optimizer_value.id!r}, 'parameters': {dict(optimizer_value.parameters)!r}}},",
                f"        'losses': {payload['losses']!r},",
                f"        'metrics': {payload['metrics']!r},",
                f"        'target_contracts': {payload['target_contracts']!r},",
                f"        'metric_target_contracts': {payload['metric_target_contracts']!r},",
                "    }",
                "",
                "",
                "def configured_losses() -> dict[str, object]:",
                f"    return {{output_id: functools.partial(_LOSS_ADAPTERS[loss_id], **parameters) for output_id, loss_id in {payload['loss_ids']!r}.items() for parameters in [{payload['loss_parameters']!r}.get(output_id, {{}})]}}",
                "",
                "",
                "def configured_metrics() -> dict[str, dict[str, object]]:",
                f"    return {{output_id: {{metric_id: functools.partial(_METRIC_ADAPTERS[metric_id], **parameters) for metric_id, parameters in zip(metric_ids, {payload['metric_parameters']!r}.get(output_id, []), strict=True)}} for output_id, metric_ids in {payload['metric_ids']!r}.items()}}",
                "",
                "",
                "def build_compilation(variables) -> dict[str, object]:",
                "    optimizer_runtime = build_optimizer()",
                "    configuration = configured_compilation()",
                "    configuration['optimizer_runtime'] = optimizer_runtime",
                "    configuration['optimizer_state'] = optimizer_runtime.init(variables)",
                "    configuration['loss_functions'] = configured_losses()",
                "    configuration['metric_functions'] = configured_metrics()",
                "    return configuration",
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
