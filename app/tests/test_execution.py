from __future__ import annotations

import time
from typing import Any, cast

import httpx
import pytest

###############################################################################
def smoke_graph(*, input_shape: list[Any] | None = None) -> dict[str, object]:
    shape = input_shape or [None, 4]
    return {
        "schema_version": 2,
        "nodes": [
            {"id": "input", "definition_id": "input.tabular", "definition_version": 1, "position": {"x": 0, "y": 0}, "parameters": {"shape": shape, "dtype": "float32"}},
            {"id": "dense", "definition_id": "layer.dense", "definition_version": 1, "position": {"x": 300, "y": 0}, "parameters": {"units": 1, "activation": "sigmoid", "use_bias": True}},
            {"id": "output", "definition_id": "output.classification", "definition_version": 1, "position": {"x": 600, "y": 0}, "parameters": {},},
        ],
        "edges": [
            {"id": "e1", "source_node_id": "input", "source_port_id": "output", "target_node_id": "dense", "target_port_id": "input"},
            {"id": "e2", "source_node_id": "dense", "source_port_id": "output", "target_node_id": "output", "target_port_id": "input"},
        ],
        "model_configuration": {},
        "export_settings": {},
        "future_training_configuration": {},
    }

###############################################################################
def worker_payload(client: httpx.Client, *, api: str = "keras", backend: str = "pytorch", device_id: str | None = None) -> dict[str, object]:
    metadata = client.get("/api/metadata").json()
    status = next(item for item in metadata["execution_backends"] if item["id"] == backend)
    device = device_id or (status.get("devices") or [{"device_id": "cpu"}])[0]["device_id"]
    return {
        "api": api,
        "backend": backend,
        "device_id": device,
        "memory_policy": {"kind": "default"},
        "precision": "float32",
    }

###############################################################################
def create_project(client: httpx.Client, graph: dict[str, object] | None = None, *, backend: str = "pytorch") -> str:
    response = client.post(
        "/api/projects",
        json={
            "name": "Runtime authority model",
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
    for _ in range(240):
        job = client.get(f"/api/execution/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "terminated", "failed"}:
            return cast(dict[str, Any], job)
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not reach a terminal state")

###############################################################################
def test_smoke_job_returns_tensor_metadata_without_training(client: httpx.Client) -> None:
    project_id = create_project(client)
    response = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "smoke_inference", "project_id": project_id, "graph": smoke_graph()}, "worker": worker_payload(client)},
    )
    assert response.status_code == 202, response.text
    job = wait_for_job(client, response.json()["id"])
    assert job["state"] == "succeeded", job
    result = cast(dict[str, Any], job["result"])
    assert result["inputs"] == [{"node_id": "input", "shape": [1, 4], "dtype": "float32"}]
    assert result["outputs"] == [{"node_id": "output", "shape": [1, 1], "dtype": "float32"}]
    assert result["parameter_count"] == 5
    assert job["pid"]
    assert job["device_lease"]["state"] == "idle"

###############################################################################
def test_smoke_job_rejects_incompatible_shape_before_worker_start(client: httpx.Client) -> None:
    project_id = create_project(client, smoke_graph(input_shape=[None, 0]))
    response = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "smoke_inference", "project_id": project_id, "graph": smoke_graph()}, "worker": worker_payload(client)},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "model_validation_failed"
    assert any(item["code"] == "invalid_shape_dimension" for item in detail["diagnostics"])

###############################################################################
def test_job_submission_reports_native_api_and_missing_backend_as_structured_errors(client: httpx.Client) -> None:
    project_id = create_project(client)
    mismatch = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "smoke_inference", "project_id": project_id, "graph": smoke_graph()}, "worker": worker_payload(client, backend="tensorflow")},
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["diagnostics"][0]["code"] == "runtime_backend_mismatch"

    native = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "smoke_inference", "project_id": project_id, "graph": smoke_graph()}, "worker": worker_payload(client, api="native")},
    )
    assert native.status_code == 422
    assert native.json()["detail"]["code"] == "model_validation_failed"
    assert native.json()["detail"]["diagnostics"][0]["code"] == "runtime_api_mismatch"

    tensorflow_project_id = create_project(client, backend="tensorflow")
    missing = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "smoke_inference", "project_id": tensorflow_project_id, "graph": smoke_graph()}, "worker": worker_payload(client, backend="tensorflow")},
    )
    if any(item["id"] == "tensorflow" and item["available"] for item in client.get("/api/metadata").json()["execution_backends"]):
        pytest.skip("TensorFlow is available in this environment")
    assert missing.status_code == 409
    assert missing.json()["detail"]["code"] == "backend_unavailable"

###############################################################################
def test_job_stop_is_idempotent_and_releases_device_lease(client: httpx.Client) -> None:
    project_id = create_project(client)
    response = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "smoke_inference", "project_id": project_id, "graph": smoke_graph()}, "worker": worker_payload(client)},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    first_stop = client.post(f"/api/execution/jobs/{job_id}/stop")
    second_stop = client.post(f"/api/execution/jobs/{job_id}/stop")
    assert first_stop.status_code == 200
    assert second_stop.status_code == 200
    terminal = wait_for_job(client, job_id)
    assert terminal["state"] in {"terminated", "succeeded"}
    replacement = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "smoke_inference", "project_id": project_id, "graph": smoke_graph()}, "worker": worker_payload(client)},
    )
    assert replacement.status_code == 202, replacement.text
    client.post(f"/api/execution/jobs/{replacement.json()['id']}/stop")
    wait_for_job(client, replacement.json()["id"])

###############################################################################
def test_health_and_metadata_are_typed_contracts(client: httpx.Client) -> None:
    health = client.get("/health")
    metadata = client.get("/api/metadata")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert metadata.status_code == 200
    assert "compilation_catalog" in metadata.json()["configuration"]
    assert metadata.json()["runtime_apis"] == [
        {"id": "keras", "available": True, "reason_code": None},
        {"id": "native", "available": False, "reason_code": "not_implemented"},
    ]
    backend_items = metadata.json()["execution_backends"]
    assert {item["id"] for item in backend_items} == {"tensorflow", "pytorch", "jax"}
    assert all(isinstance(item["installed"], bool) for item in backend_items)
    assert all("devices" in item for item in backend_items)
