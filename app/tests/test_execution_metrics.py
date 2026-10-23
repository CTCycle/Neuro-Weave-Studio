# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from common.process_control import process_start_options
from domain.metric_formula import MetricFormulaError, emit_metric_formula, parse_metric_formula
from services.runtime_controller import RuntimeController

# Consolidated from test_execution.py

###############################################################################
def smoke_graph(*, input_shape: list[Any] | None = None) -> dict[str, object]:
    shape = input_shape or [4]
    return {
        "schema_version": 6,
        "nodes": [
            {
                "id": "input",
                "definition_id": "input.tabular",
                "definition_version": 1,
                "position": {"x": 0, "y": 0},
                "parameters": {"shape": shape, "dtype": "float32"},
            },
            {
                "id": "dense",
                "definition_id": "layer.dense",
                "definition_version": 1,
                "position": {"x": 300, "y": 0},
                "parameters": {
                    "units": 1,
                    "activation": "sigmoid",
                    "use_bias": True,
                },
            },
            {
                "id": "output",
                "definition_id": "output.classification",
                "definition_version": 1,
                "position": {"x": 600, "y": 0},
                "parameters": {},
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source_node_id": "input",
                "source_port_id": "output",
                "target_node_id": "dense",
                "target_port_id": "input",
            },
            {
                "id": "e2",
                "source_node_id": "dense",
                "source_port_id": "output",
                "target_node_id": "output",
                "target_port_id": "input",
            },
        ],
        "model_configuration": {},
        "export_settings": {},
        "future_training_configuration": {},
    }

###############################################################################
def worker_payload(
    client: httpx.Client,
    *,
    api: str = "keras",
    backend: str = "pytorch",
    require_operational: bool = False,
) -> dict[str, object]:
    metadata = client.get("/api/metadata").json()
    status = next(item for item in metadata["execution_backends"] if item["id"] == backend)
    if require_operational and not status["installed"]:
        pytest.skip(f"{backend} backend is not installed")
    discovery = client.post(
        "/api/execution/discovery",
        json={"backend": backend, "api": api},
    )
    assert discovery.status_code == 202, discovery.text
    # A cold PyTorch/Keras probe can spend more than ten seconds importing
    # torch and Keras. Keep the test bounded, but align the polling window
    # with Settings.runtime_discovery_timeout_seconds (the service limit is
    # 60 seconds) instead of racing the first cold import.
    for _ in range(1_200):
        current = client.get(
            "/api/execution/discovery",
            params={"backend": backend, "api": api},
        ).json()
        if current["state"] != "checking":
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"{backend}/{api} discovery did not finish")
    if require_operational and not current["available"]:
        pytest.skip(f"Operational {backend}/{api} runtime is unavailable")
    metadata = client.get("/api/metadata").json()
    status = next(item for item in metadata["execution_backends"] if item["id"] == backend)
    device = next(
        (item for item in status.get("devices", []) if item.get("available", True)),
        None,
    )
    device_id = device["device_id"] if device is not None else "cpu"
    return {
        "api": api,
        "backend": backend,
        "device_id": device_id,
        "memory_policy": {"kind": "default"},
        "precision": "float32",
    }

###############################################################################
def create_project(
    client: httpx.Client,
    graph: dict[str, object] | None = None,
    *,
    backend: str = "pytorch",
) -> str:
    response = client.post(
        "/api/projects",
        json={
            "name": "Runtime model",
            "description": "",
            "execution_backend": backend,
            "keras_enabled": True,
            "graph": graph or smoke_graph(),
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])

###############################################################################
def wait_for_job(client: httpx.Client, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        job = client.get(f"/api/execution/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "terminated", "failed"}:
            return cast(dict[str, Any], job)
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not reach a terminal state")

###############################################################################
def fail_runtime_import(job_dir: Path) -> subprocess.Popen[bytes]:
    log_handle = (job_dir / "worker.log").open("wb")
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", "import neuro_weave_missing_test_runtime"],
            cwd=job_dir,
            stdout=log_handle,
            stderr=log_handle,
            **process_start_options(),
        )
    finally:
        log_handle.close()
    return cast(subprocess.Popen[bytes], process)

###############################################################################
def test_smoke_job_returns_tensor_metadata_without_training(
    client: httpx.Client,
) -> None:
    project_id = create_project(client)
    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "smoke_inference",
                "project_id": project_id,
                "graph": smoke_graph(),
            },
            "worker": worker_payload(client, require_operational=True),
        },
    )
    assert response.status_code == 202, response.text

    job = wait_for_job(client, response.json()["id"])
    assert job["state"] == "succeeded", job
    result = cast(dict[str, Any], job["result"])
    assert result["inputs"] == [{"node_id": "input", "shape": [1, 4], "dtype": "float32"}]
    assert result["outputs"] == [{"node_id": "output", "shape": [1, 1], "dtype": "float32"}]
    assert result["parameter_count"] == 5

###############################################################################
def test_controlled_worker_import_failure_preserves_health_and_recovers(
    client: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_app = cast(Any, cast(TestClient, client).app)
    runtime = cast(RuntimeController, test_app.state.runtime_controller)
    original_start = cast(
        Callable[[Path], subprocess.Popen[bytes]],
        runtime._start_process,  # pyright: ignore[reportPrivateUsage]
    )
    monkeypatch.setattr(runtime, "_start_process", fail_runtime_import)

    project_id = create_project(client)
    worker = worker_payload(client, require_operational=True)
    assert worker["backend"] == "pytorch"
    assert worker["api"] == "keras"
    operation = {
        "type": "smoke_inference",
        "project_id": project_id,
        "graph": smoke_graph(),
    }

    failed_response = client.post(
        "/api/execution/jobs",
        json={"operation": operation, "worker": worker},
    )
    assert failed_response.status_code == 202, failed_response.text
    failed = wait_for_job(client, failed_response.json()["id"])
    assert failed["state"] == "failed"
    assert failed["worker"]["backend"] == "pytorch"
    assert failed["worker"]["api"] == "keras"
    assert failed["error"]["code"] == "worker_failed"
    assert "neuro_weave_missing_test_runtime" in failed["error"]["message"]
    assert failed["device_lease"]["state"] == "failed"
    assert client.get("/health").json() == {"status": "ok"}

    job_root = tmp_path / "jobs"
    for _ in range(50):
        if not job_root.exists() or not any(job_root.iterdir()):
            break
        time.sleep(0.02)
    assert not job_root.exists() or not any(job_root.iterdir())

    monkeypatch.setattr(runtime, "_start_process", original_start)
    recovered_response = client.post(
        "/api/execution/jobs",
        json={"operation": operation, "worker": worker},
    )
    assert recovered_response.status_code == 202, recovered_response.text
    recovered = wait_for_job(client, recovered_response.json()["id"])
    assert recovered["state"] == "succeeded", recovered
    assert recovered["worker"]["backend"] == "pytorch"
    assert recovered["worker"]["api"] == "keras"
    result = cast(dict[str, Any], recovered["result"])
    assert result["outputs"] == [{"node_id": "output", "shape": [1, 1], "dtype": "float32"}]
    assert client.get("/health").json() == {"status": "ok"}

###############################################################################
def test_smoke_job_rejects_invalid_shape_before_worker_start(
    client: httpx.Client,
) -> None:
    invalid_graph = smoke_graph(input_shape=[0])
    project_id = create_project(client, invalid_graph)

    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "smoke_inference",
                "project_id": project_id,
                "graph": invalid_graph,
            },
            "worker": worker_payload(client),
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "model_validation_failed"
    assert any(item["code"] == "invalid_shape_dimension" for item in detail["diagnostics"])

###############################################################################
def test_job_submission_rejects_runtime_selection_inconsistent_with_model(
    client: httpx.Client,
) -> None:
    project_id = create_project(client)

    backend_mismatch = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "smoke_inference",
                "project_id": project_id,
                "graph": smoke_graph(),
            },
            "worker": worker_payload(client, backend="tensorflow"),
        },
    )
    assert backend_mismatch.status_code == 422
    assert backend_mismatch.json()["detail"]["diagnostics"][0]["code"] == "runtime_backend_mismatch"

    native = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "smoke_inference",
                "project_id": project_id,
                "graph": smoke_graph(),
            },
            "worker": worker_payload(client, api="native"),
        },
    )
    assert native.status_code == 422
    assert native.json()["detail"]["diagnostics"][0]["code"] == "runtime_api_mismatch"

###############################################################################
def test_job_stop_is_idempotent_and_releases_device_lease(
    client: httpx.Client,
) -> None:
    project_id = create_project(client)
    worker = worker_payload(client, require_operational=True)
    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "smoke_inference",
                "project_id": project_id,
                "graph": smoke_graph(),
            },
            "worker": worker,
        },
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]

    assert client.post(f"/api/execution/jobs/{job_id}/stop").status_code == 200
    assert client.post(f"/api/execution/jobs/{job_id}/stop").status_code == 200
    terminal = wait_for_job(client, job_id)
    assert terminal["state"] in {"terminated", "succeeded"}
    assert terminal["device_lease"]["state"] == "idle"

    replacement = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "smoke_inference",
                "project_id": project_id,
                "graph": smoke_graph(),
            },
            "worker": worker,
        },
    )
    assert replacement.status_code == 202, replacement.text
    replacement_id = replacement.json()["id"]
    client.post(f"/api/execution/jobs/{replacement_id}/stop")
    wait_for_job(client, replacement_id)

# Consolidated from test_metrics.py

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
