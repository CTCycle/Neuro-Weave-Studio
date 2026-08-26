from __future__ import annotations

import time
from typing import Any, cast

import httpx
import pytest

from repositories.graphs import GraphRepository


###############################################################################
def fail_update_graph_replace(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("graph replacement failed")


###############################################################################
def project_payload() -> dict[str, object]:
    return {
        "name": "API MLP",
        "description": "Persistence integration test",
        "execution_backend": "pytorch",
        "keras_enabled": True,
        "graph": {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "input",
                    "definition_id": "input.tabular",
                    "definition_version": 1,
                    "position": {"x": 0, "y": 0},
                    "parameters": {"shape": [None, 8], "dtype": "float32"},
                },
                {
                    "id": "dense",
                    "definition_id": "layer.dense",
                    "definition_version": 1,
                    "position": {"x": 300, "y": 0},
                    "parameters": {
                        "units": 2,
                        "activation": "softmax",
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
            "model_configuration": {
                "optimizer": "adam",
                "optimizer_parameters": {"learning_rate": 0.001},
                "losses": {},
                "metrics": {},
                "mixed_precision": False,
                "random_seed": 42,
            },
            "export_settings": {
                "include_comments": True,
                "include_shape_assertions": True,
                "preferred_binary_format": "keras",
            },
            "future_training_configuration": {},
        },
    }


###############################################################################
def execution_worker(client: httpx.Client) -> dict[str, object]:
    metadata = client.get("/api/metadata").json()
    status = next(
        item for item in metadata["execution_backends"] if item["id"] == "pytorch"
    )
    if not status["available"]:
        pytest.skip("Operational PyTorch Keras backend is unavailable")
    device = next(
        (item for item in status.get("devices", []) if item.get("available", True)),
        None,
    )
    if device is None:
        pytest.skip("No operational PyTorch device is available")
    return {
        "api": "keras",
        "backend": "pytorch",
        "device_id": device["device_id"],
        "memory_policy": {"kind": "default"},
        "precision": "float32",
    }


###############################################################################
def test_project_revision_workflow_invalidates_generated_source_and_rejects_stale_save(
    client: httpx.Client,
) -> None:
    created = client.post("/api/projects", json=project_payload())
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["current_revision"] == 1
    assert project["generated_source"] is None

    reopened = client.get(f"/api/projects/{project['id']}")
    assert reopened.status_code == 200
    assert reopened.json()["graph"]["nodes"][1]["definition_id"] == "layer.dense"

    validation = client.post(
        "/api/validation",
        json={
            "graph": project["graph"],
            "execution_backend": "pytorch",
            "keras_enabled": True,
        },
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    keras_code = client.post(
        "/api/compilation",
        json={
            "graph": project["graph"],
            "execution_backend": "pytorch",
            "keras_enabled": True,
        },
    )
    native_code = client.post(
        "/api/compilation",
        json={
            "graph": project["graph"],
            "execution_backend": "pytorch",
            "keras_enabled": False,
        },
    )
    assert "keras.Model" in keras_code.json()["code"]
    assert "class GeneratedModel" in native_code.json()["code"]

    source = client.get(f"/api/exports/{project['id']}/source")
    assert source.status_code == 200
    assert client.get(f"/api/projects/{project['id']}").json()["generated_source"] is not None

    update_payload = project_payload()
    update_payload["name"] = "API MLP updated"
    update_payload["expected_revision"] = project["current_revision"]
    updated = client.put(f"/api/projects/{project['id']}", json=update_payload)
    assert updated.status_code == 200
    assert updated.json()["current_revision"] == 2
    assert updated.json()["generated_source"] is None

    stale = client.put(f"/api/projects/{project['id']}", json=update_payload)
    assert stale.status_code == 409

    missing_revision = project_payload()
    assert (
        client.put(f"/api/projects/{project['id']}", json=missing_revision).status_code
        == 422
    )


###############################################################################
def test_keras_export_publishes_only_verified_artifact(client: httpx.Client) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {"type": "keras_export", "project_id": project["id"]},
            "worker": execution_worker(client),
        },
    )
    assert response.status_code == 202, response.text

    job_id = response.json()["id"]
    job: dict[str, Any] | None = None
    for _ in range(240):
        job = cast(
            dict[str, Any],
            client.get(f"/api/execution/jobs/{job_id}").json(),
        )
        if job["state"] in {"succeeded", "terminated", "failed"}:
            break
        time.sleep(0.05)

    assert job is not None
    assert job["state"] == "succeeded", job
    assert job["artifact_available"] is True

    artifact = client.get(f"/api/execution/jobs/{job_id}/artifact")
    assert artifact.status_code == 200
    assert artifact.content.startswith(b"PK")


###############################################################################
def test_keras_export_rejects_invalid_graph_before_worker_start(
    client: httpx.Client,
) -> None:
    payload = project_payload()
    graph = cast(dict[str, Any], payload["graph"])
    nodes = cast(list[dict[str, Any]], graph["nodes"])
    nodes[1]["definition_id"] = "unknown.node"
    project = client.post("/api/projects", json=payload).json()

    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {"type": "keras_export", "project_id": project["id"]},
            "worker": {
                "api": "keras",
                "backend": "pytorch",
                "device_id": "cpu",
                "memory_policy": {"kind": "default"},
                "precision": "float32",
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "model_validation_failed"


###############################################################################
def test_project_json_export_import_preserves_editable_graph(
    client: httpx.Client,
) -> None:
    project = client.post("/api/projects", json=project_payload()).json()

    bundle = client.get(f"/api/projects/{project['id']}/export")
    assert bundle.status_code == 200
    imported = client.post("/api/projects/import", json=bundle.json())

    assert imported.status_code == 201
    assert imported.json()["name"] == "API MLP"
    assert imported.json()["graph"] == project["graph"]


###############################################################################
def test_update_rolls_back_metadata_and_revision_when_graph_persistence_fails(
    client: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = client.post("/api/projects", json=project_payload()).json()

    monkeypatch.setattr(GraphRepository, "replace", fail_update_graph_replace)
    update_payload = project_payload()
    update_payload["name"] = "Should roll back"
    update_payload["expected_revision"] = created["current_revision"]

    with pytest.raises(RuntimeError, match="graph replacement failed"):
        client.put(f"/api/projects/{created['id']}", json=update_payload)

    unchanged = client.get(f"/api/projects/{created['id']}")
    assert unchanged.status_code == 200
    assert unchanged.json()["name"] == created["name"]
    assert unchanged.json()["current_revision"] == 1
