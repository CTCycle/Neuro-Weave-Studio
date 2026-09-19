from __future__ import annotations

import httpx
import pytest

from domain.metric_formula import MetricFormulaError, emit_metric_formula, parse_metric_formula

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
    pytest.importorskip("keras")
    import keras

    from models.keras_runtime import PredictionAggregateMetric

    target = keras.ops.zeros((2, 1), dtype="float32")
    prediction = keras.ops.convert_to_tensor([[1.0], [3.0]], dtype="float32")
    mean = PredictionAggregateMetric("mean")
    total = PredictionAggregateMetric("sum")
    mean.update_state(target, prediction)
    total.update_state(target, prediction)

    assert float(keras.ops.convert_to_numpy(mean.result())) == pytest.approx(2.0)
    assert float(keras.ops.convert_to_numpy(total.result())) == pytest.approx(4.0)

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

    project = {
        "name": "Metric reference model",
        "description": "Metric reference test",
        "execution_backend": "pytorch",
        "keras_enabled": True,
        "graph": {
            "schema_version": 5,
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
                    "optimizer": "adam",
                    "optimizer_parameters": {"learning_rate": 0.001},
                    "losses": {"__default__": "mean_squared_error"},
                    "metrics": {"__default__": [metric_id]},
                    "mixed_precision": False,
                    "random_seed": 42,
                },
                "native": {},
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
