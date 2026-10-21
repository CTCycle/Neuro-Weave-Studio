# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

from domain.metric_formula import (
    MetricFormulaError,
    emit_metric_formula,
    parse_metric_formula,
)


###############################################################################
def test_metric_formula_uses_only_the_bounded_expression_subset() -> None:
    formula = r"\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2"
    node = parse_metric_formula(formula)

    assert "keras.ops.mean" in emit_metric_formula(node, "keras")
    assert "keras.ops.square" in emit_metric_formula(node, "keras")
    with pytest.raises(MetricFormulaError):
        parse_metric_formula("__import__('os').system('whoami')")

###############################################################################
def test_prediction_aggregate_metrics_use_prediction_values() -> None:
    keras: Any = pytest.importorskip("keras")

    from models.keras_runtime import PredictionAggregateMetric

    target = keras.ops.zeros((2, 1), dtype="float32")
    prediction = keras.ops.convert_to_tensor([[1.0], [3.0]], dtype="float32")
    mean = PredictionAggregateMetric("mean")
    total = PredictionAggregateMetric("sum")
    mean.update_state(target, prediction)
    total.update_state(target, prediction)

    approx = cast(Callable[[object], object], pytest.approx)  # pyright: ignore[reportUnknownMemberType]
    assert float(keras.ops.convert_to_numpy(mean.result())) == approx(2.0)
    assert float(keras.ops.convert_to_numpy(total.result())) == approx(4.0)

###############################################################################
def test_custom_metric_lifecycle_and_builtin_locking(client: httpx.Client) -> None:
    builtins = client.get("/api/metrics")
    assert builtins.status_code == 200
    builtin = next(item for item in builtins.json()["definitions"] if item["id"] == "mean_squared_error")
    assert builtin["locked"] is True
    assert builtin["can_edit"] is False
    assert builtin["can_delete"] is False

    created = client.post(
        "/api/metrics",
        json={
            "kind": "metric",
            "label": "Readable error",
            "description": "A custom squared error metric.",
            "formula": r"\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    metric_id = definition["id"]
    assert metric_id.startswith("custom-")
    assert definition["locked"] is False
    assert definition["can_edit"] is True
    assert definition["can_delete"] is True
    assert definition["supported_tasks"] == [
        "regression",
        "binary_classification",
        "multiclass_classification",
    ]
    assert definition["capabilities"] == {"keras": True, "pytorch": True}

    duplicate = client.post(
        "/api/metrics",
        json={
            "kind": "metric",
            "label": "  READABLE ERROR ",
            "description": "Duplicate.",
            "formula": "mean((y-yhat)^2)",
        },
    )
    assert duplicate.status_code == 409

    updated = client.put(
        f"/api/metrics/{metric_id}",
        json={
            "label": "Readable error v2",
            "description": "The edited custom metric.",
            "formula": "mean(abs(y-yhat))",
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["id"] == metric_id
    assert updated.json()["formula"] == "mean(abs(y-yhat))"
    reloaded = next(
        item for item in client.get("/api/metrics").json()["definitions"] if item["id"] == metric_id
    )
    assert reloaded["label"] == "Readable error v2"
    assert reloaded["formula"] == "mean(abs(y-yhat))"

    locked_update = client.put(
        "/api/metrics/mean_squared_error",
        json={"label": "Changed", "description": "No", "formula": "y-yhat"},
    )
    assert locked_update.status_code == 403
    locked_delete = client.delete("/api/metrics/mean_squared_error")
    assert locked_delete.status_code == 403

    deleted = client.delete(f"/api/metrics/{metric_id}")
    assert deleted.status_code == 204
    assert all(item["id"] != metric_id for item in client.get("/api/metrics").json()["definitions"])

###############################################################################
def test_custom_metric_and_loss_persist_into_supported_compilation_catalogs(
    client: httpx.Client,
) -> None:
    created_metric = client.post(
        "/api/metrics",
        json={
            "kind": "metric",
            "label": "Persistent validation metric",
            "description": "Metric catalog persistence test.",
            "formula": "mean(abs(y-yhat))",
        },
    )
    created_loss = client.post(
        "/api/metrics",
        json={
            "kind": "loss",
            "label": "Persistent validation loss",
            "description": "Loss catalog persistence test.",
            "formula": "mean((y-yhat)^2)",
        },
    )
    assert created_metric.status_code == 201, created_metric.text
    assert created_loss.status_code == 201, created_loss.text
    metric_id = created_metric.json()["id"]
    loss_id = created_loss.json()["id"]

    try:
        definitions = client.get("/api/metrics").json()["definitions"]
        listed = {item["id"]: item for item in definitions}
        assert listed[metric_id]["kind"] == "metric"
        assert listed[loss_id]["kind"] == "loss"
        assert listed[metric_id]["capabilities"] == {"keras": True, "pytorch": True}
        assert listed[loss_id]["capabilities"] == {"keras": True, "pytorch": True}

        capabilities = client.get("/api/metadata").json()["configuration"]["compilation_capabilities"]
        keras_metric_ids = {item["id"] for item in capabilities["keras"]["metrics"]}
        keras_loss_ids = {item["id"] for item in capabilities["keras"]["losses"]}
        pytorch_catalog = capabilities["native"]["pytorch"]["catalog"]
        pytorch_metric_ids = {item["id"] for item in pytorch_catalog["metrics"]}
        pytorch_loss_ids = {item["id"] for item in pytorch_catalog["losses"]}
        assert metric_id in keras_metric_ids
        assert loss_id in keras_loss_ids
        assert metric_id in pytorch_metric_ids
        assert loss_id in pytorch_loss_ids

        for backend in ("tensorflow", "jax"):
            native_catalog = capabilities["native"][backend]["catalog"]
            assert metric_id not in {item["id"] for item in native_catalog["metrics"]}
            assert loss_id not in {item["id"] for item in native_catalog["losses"]}
    finally:
        assert client.delete(f"/api/metrics/{metric_id}").status_code == 204
        assert client.delete(f"/api/metrics/{loss_id}").status_code == 204

###############################################################################
def test_custom_metric_formula_validation_rejects_unsafe_input(client: httpx.Client) -> None:
    rejected = client.post(
        "/api/metrics",
        json={
            "kind": "metric",
            "label": "Unsafe formula",
            "description": "Must be rejected.",
            "formula": "__import__('os').system('whoami')",
        },
    )
    assert rejected.status_code == 422

###############################################################################
def test_custom_metric_delete_reports_saved_model_references(client: httpx.Client) -> None:
    created = client.post(
        "/api/metrics",
        json={
            "kind": "metric",
            "label": "Referenced metric",
            "description": "Used by the fixture project.",
            "formula": "mean(abs(y-yhat))",
        },
    )
    assert created.status_code == 201, created.text
    metric_id = created.json()["id"]

    project: dict[str, Any] = {
        "name": "Metric reference model",
        "description": "Metric reference test",
        "execution_backend": "pytorch",
        "keras_enabled": True,
        "graph": {
            "schema_version": 6,
            "nodes": [
                {"id": "input", "definition_id": "input.tabular", "position": {"x": 0, "y": 0}, "parameters": {"shape": [4], "dtype": "float32"}},
                {"id": "dense", "definition_id": "layer.dense", "position": {"x": 300, "y": 0}, "parameters": {"units": 1, "activation": "linear", "use_bias": True}},
                {"id": "output", "definition_id": "output.regression", "position": {"x": 600, "y": 0}, "parameters": {}},
            ],
            "edges": [
                {"id": "e1", "source_node_id": "input", "source_port_id": "output", "target_node_id": "dense", "target_port_id": "input"},
                {"id": "e2", "source_node_id": "dense", "source_port_id": "output", "target_node_id": "output", "target_port_id": "input"},
            ],
            "model_configuration": {
                "keras": {
                    "compilation": {
                        "optimizer": {
                            "id": "adam",
                            "parameters": {"learning_rate": 0.001},
                        },
                        "losses": {
                            "__default__": {
                                "id": "mean_squared_error",
                                "parameters": {},
                            }
                        },
                        "metrics": {
                            "__default__": [{"id": metric_id, "parameters": {}}]
                        },
                    },
                    "mixed_precision": False,
                    "random_seed": 42,
                }
            },
            "export_settings": {"include_comments": True, "include_shape_assertions": True, "preferred_binary_format": "keras"},
            "future_training_configuration": {},
        },
    }
    project_response = client.post("/api/projects", json=project)
    assert project_response.status_code == 201, project_response.text

    blocked = client.delete(f"/api/metrics/{metric_id}")
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["detail"]["usages"][0]["project_name"] == "Metric reference model"
