from __future__ import annotations

import subprocess
from typing import Any, cast

import httpx
import pytest

from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from models.keras_backend_runner import KerasBackendRunner

###############################################################################
def smoke_graph(*, input_shape: list[Any] | None = None) -> dict[str, object]:
    shape = input_shape or [None, 4]
    return {
        "schema_version": 2,
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
                "parameters": {"units": 1, "activation": "sigmoid", "use_bias": True},
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
def test_smoke_test_returns_tensor_metadata_without_training(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    cast(Any, client).app.state.backend_statuses = {
        ExecutionBackend.PYTORCH: {
            "available": True,
            "accelerator": "cuda",
            "device_name": "Test GPU",
        }
    }
    def fake_smoke(_graph: ModelGraph, _backend: ExecutionBackend) -> dict[str, object]:
        return {
            "accelerator": "cuda",
            "device_name": "Test GPU",
            "inputs": [{"node_id": "input", "shape": [1, 4], "dtype": "float32"}],
            "outputs": [{"node_id": "output", "shape": [1, 1], "dtype": "float32"}],
            "parameter_count": 5,
            "duration_ms": 2.5,
        }

    monkeypatch.setattr(cast(Any, client).app.state.backend_runner, "smoke_test", fake_smoke)

    response = client.post(
        "/api/execution/smoke-test",
        json={"graph": smoke_graph(), "execution_backend": "pytorch"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "status": "succeeded",
        "execution_backend": "pytorch",
        "accelerator": "cuda",
        "device_name": "Test GPU",
        "inputs": [{"node_id": "input", "shape": [1, 4], "dtype": "float32"}],
        "outputs": [{"node_id": "output", "shape": [1, 1], "dtype": "float32"}],
        "parameter_count": 5,
        "duration_ms": 2.5,
    }

###############################################################################
def test_smoke_test_rejects_incompatible_shape_with_actionable_diagnostics(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    cast(Any, client).app.state.backend_statuses = {
        ExecutionBackend.PYTORCH: {"available": True}
    }
    worker_called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        nonlocal worker_called
        worker_called = True

    monkeypatch.setattr(cast(Any, client).app.state.backend_runner, "smoke_test", fail_if_called)
    response = client.post(
        "/api/execution/smoke-test",
        json={
            "graph": smoke_graph(input_shape=[None, 0]),
            "execution_backend": "pytorch",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "model_validation_failed"
    assert any(
        item["code"] == "invalid_shape_dimension"
        for item in response.json()["detail"]["diagnostics"]
    )
    assert worker_called is False

###############################################################################
def test_smoke_test_reports_missing_backend_without_crashing(client: httpx.Client) -> None:
    cast(Any, client).app.state.backend_statuses = {
        ExecutionBackend.TENSORFLOW: {
            "available": False,
            "reason_code": "dependency_missing",
        }
    }
    response = client.post(
        "/api/execution/smoke-test",
        json={"graph": smoke_graph(), "execution_backend": "tensorflow"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "backend_unavailable",
        "backend": "tensorflow",
        "reason_code": "dependency_missing",
        "message": "Execution backend 'tensorflow' is not operational",
    }

###############################################################################
def test_smoke_test_reports_worker_timeout_without_crashing(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    cast(Any, client).app.state.backend_statuses = {
        ExecutionBackend.PYTORCH: {"available": True}
    }

    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="keras_worker", timeout=300)

    monkeypatch.setattr(cast(Any, client).app.state.backend_runner, "smoke_test", timeout)
    response = client.post(
        "/api/execution/smoke-test",
        json={"graph": smoke_graph(), "execution_backend": "pytorch"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "worker_failed",
        "message": "Command 'keras_worker' timed out after 300 seconds",
    }

###############################################################################
def test_worker_smoke_test_builds_and_executes_minimal_model() -> None:
    runner = KerasBackendRunner()
    backend = next(iter(runner.available_backends()), None)
    if backend is None:
        pytest.skip("No operational Keras backend is installed")

    result = runner.smoke_test(ModelGraph.model_validate(smoke_graph()), backend)

    assert result["status"] == "succeeded"
    assert result["inputs"] == [{"node_id": "input", "shape": [1, 4], "dtype": "float32"}]
    assert result["outputs"] == [{"node_id": "output", "shape": [1, 1], "dtype": "float32"}]
    assert result["parameter_count"] == 5
