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
    status = next(
        item for item in metadata["execution_backends"] if item["id"] == backend
    )
    if require_operational and not status["available"]:
        pytest.skip(f"Operational {backend} backend is unavailable")
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
    assert result["inputs"] == [
        {"node_id": "input", "shape": [1, 4], "dtype": "float32"}
    ]
    assert result["outputs"] == [
        {"node_id": "output", "shape": [1, 1], "dtype": "float32"}
    ]
    assert result["parameter_count"] == 5


###############################################################################
def test_smoke_job_rejects_invalid_shape_before_worker_start(
    client: httpx.Client,
) -> None:
    invalid_graph = smoke_graph(input_shape=[None, 0])
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
    assert any(
        item["code"] == "invalid_shape_dimension" for item in detail["diagnostics"]
    )


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
    assert (
        backend_mismatch.json()["detail"]["diagnostics"][0]["code"]
        == "runtime_backend_mismatch"
    )

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
