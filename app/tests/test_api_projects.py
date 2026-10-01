# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from app import create_app
from repositories.graphs import GraphRepository


###############################################################################
def fail_update_graph_replace(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("graph replacement failed")

###############################################################################
def ensure_discovery(client: httpx.Client, backend: str, api: str) -> dict[str, object]:
    response = client.post(
        "/api/execution/discovery",
        json={"backend": backend, "api": api},
    )
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        current = client.get(
            "/api/execution/discovery",
            params={"backend": backend, "api": api},
        ).json()
        if current["state"] != "checking":
            return cast(dict[str, object], current)
        time.sleep(0.05)
    pytest.fail(f"{backend}/{api} discovery did not finish")

###############################################################################
def project_payload() -> dict[str, object]:
    return {
        "name": "API MLP",
        "description": "Persistence integration test",
        "execution_backend": "pytorch",
        "keras_enabled": True,
        "graph": {
            "schema_version": 5,
            "nodes": [
                {
                    "id": "input",
                    "definition_id": "input.tabular",
                    "definition_version": 1,
                    "position": {"x": 0, "y": 0},
                    "parameters": {"shape": [8], "dtype": "float32"},
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
                "keras": {
                    "optimizer": "adam",
                    "optimizer_parameters": {"learning_rate": 0.001},
                    "losses": {},
                    "metrics": {},
                    "mixed_precision": False,
                    "random_seed": 42,
                },
                "native": {},
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
    status = next(item for item in metadata["execution_backends"] if item["id"] == "pytorch")
    if not status["installed"]:
        pytest.skip("PyTorch backend is not installed")
    discovered = ensure_discovery(client, "pytorch", "keras")
    if not discovered["available"]:
        pytest.skip("Operational PyTorch Keras backend is unavailable")
    metadata = client.get("/api/metadata").json()
    status = next(item for item in metadata["execution_backends"] if item["id"] == "pytorch")
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
def native_execution_worker(client: httpx.Client) -> dict[str, object]:
    metadata = client.get("/api/metadata").json()
    status = next(item for item in metadata["execution_backends"] if item["id"] == "pytorch")
    if not status["installed"]:
        pytest.skip("PyTorch backend is not installed")
    ensure_discovery(client, "pytorch", "native")
    metadata = client.get("/api/metadata").json()
    status = next(item for item in metadata["execution_backends"] if item["id"] == "pytorch")
    native_status = next(item for item in status.get("runtime_apis", []) if item["id"] == "native")
    if not native_status["available"]:
        pytest.skip("Operational native PyTorch runtime is unavailable")
    device = next(
        (item for item in status.get("devices", []) if item.get("available", True)),
        None,
    )
    if device is None:
        pytest.skip("No operational PyTorch device is available")
    return {
        "api": "native",
        "backend": "pytorch",
        "device_id": device["device_id"],
        "memory_policy": {"kind": "default"},
        "precision": "float32",
    }

###############################################################################
def wait_for_job(client: httpx.Client, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        job = cast(dict[str, Any], client.get(f"/api/execution/jobs/{job_id}").json())
        if job["state"] in {"succeeded", "terminated", "failed"}:
            return job
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not reach a terminal state")

###############################################################################
def test_project_revision_workflow_invalidates_generated_source_and_rejects_stale_save(
    client: httpx.Client,
) -> None:
    created = client.post("/api/projects", json=project_payload())
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["current_revision"] == 1
    assert project["generated_source"] is None
    assert "schema_version" not in project
    assert project["graph"]["schema_version"] == 6
    assert project["graph"]["nodes"][0]["parameters"]["shape"] == [8]

    listed = client.get("/api/projects")
    assert listed.status_code == 200
    summary = next(item for item in listed.json()["projects"] if item["id"] == project["id"])
    assert summary["input_count"] == 1
    assert summary["output_count"] == 1

    reopened = client.get(f"/api/projects/{project['id']}")
    assert reopened.status_code == 200
    assert reopened.json()["graph"]["nodes"][1]["definition_id"] == "layer.dense"
    assert reopened.json()["graph"]["nodes"][0]["parameters"]["shape"] == [8]

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
    assert client.put(f"/api/projects/{project['id']}", json=missing_revision).status_code == 422

###############################################################################
def test_project_revision_survives_application_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "project-restart.db"
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(database))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("NEURO_WEAVE_ML_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("NEURO_WEAVE_RUNTIME_DISCOVERY_PATH", str(tmp_path / "environment.json"))

    with TestClient(create_app()) as first:
        created_response = first.post("/api/projects", json=project_payload())
        assert created_response.status_code == 201, created_response.text
        created = created_response.json()

        update_payload = project_payload()
        update_payload["name"] = "API MLP persisted"
        update_payload["expected_revision"] = created["current_revision"]
        updated_response = first.put(f"/api/projects/{created['id']}", json=update_payload)
        assert updated_response.status_code == 200, updated_response.text
        saved = updated_response.json()
        assert saved["current_revision"] == 2

    with TestClient(create_app()) as second:
        reopened_response = second.get(f"/api/projects/{created['id']}")
        assert reopened_response.status_code == 200
        reopened = reopened_response.json()
        assert reopened["name"] == "API MLP persisted"
        assert reopened["current_revision"] == 2
        assert reopened["execution_backend"] == saved["execution_backend"]
        assert reopened["keras_enabled"] == saved["keras_enabled"]
        assert reopened["graph"] == saved["graph"]

        listed = second.get("/api/projects")
        assert listed.status_code == 200
        assert any(item["id"] == created["id"] for item in listed.json()["projects"])

###############################################################################
def test_model_export_publishes_only_verified_artifact(client: httpx.Client) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {"type": "model_export", "project_id": project["id"], "format": "keras"},
            "worker": execution_worker(client),
        },
    )
    assert response.status_code == 202, response.text

    job_id = response.json()["id"]
    job = wait_for_job(client, job_id)
    assert job["state"] == "succeeded", job
    assert job["artifact"] is not None
    assert job["artifact"]["format"] == "keras"

    artifact = client.get(f"/api/execution/jobs/{job_id}/artifact")
    assert artifact.status_code == 200
    assert artifact.content.startswith(b"PK")

###############################################################################
def test_model_export_rejects_invalid_graph_before_worker_start(
    client: httpx.Client,
    tmp_path: Path,
) -> None:
    payload = project_payload()
    graph = cast(dict[str, Any], payload["graph"])
    nodes = cast(list[dict[str, Any]], graph["nodes"])
    nodes[1]["definition_id"] = "unknown.node"
    project = client.post("/api/projects", json=payload).json()

    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {"type": "model_export", "project_id": project["id"], "format": "keras"},
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
    assert not list((tmp_path / "exports").rglob("*"))

###############################################################################
def test_model_export_rejects_worker_profile_mismatch_before_job_creation(
    client: httpx.Client,
) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    worker = {
        "api": "keras",
        "backend": "tensorflow",
        "device_id": "cpu",
        "memory_policy": {"kind": "default"},
        "precision": "float32",
    }

    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {"type": "model_export", "project_id": project["id"], "format": "keras"},
            "worker": worker,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "model_validation_failed"
    assert detail["diagnostics"][0]["code"] == "runtime_backend_mismatch"
    assert client.get("/api/execution/jobs", params={"active_only": True}).json() == []

###############################################################################
def test_export_capabilities_expose_matrix_and_explicit_reasons(
    client: httpx.Client,
) -> None:
    project = client.post("/api/projects", json=project_payload()).json()

    response = client.get(f"/api/exports/{project['id']}/capabilities")

    assert response.status_code == 200, response.text
    payload = response.json()
    formats = {item["format"]: item for item in payload["formats"]}
    assert "torchscript" not in formats
    assert payload["project_revision"] == project["current_revision"]
    assert formats["project_bundle"]["available"] is True
    assert formats["source"]["available"] is True
    assert formats["keras"]["available"] is False
    assert formats["keras"]["reason_code"] == "not_checked"
    assert formats["pytorch_exported_program"]["available"] is False
    assert formats["pytorch_exported_program"]["reason_code"] == "keras_enabled"
    assert {item["id"] for item in formats["onnx"]["options"]} >= {
        "filename_stem",
        "opset_version",
        "dynamic_batch",
        "external_data",
    }
    opset = next(item for item in formats["onnx"]["options"] if item["id"] == "opset_version")
    assert opset["choices"] == [17, 18]

###############################################################################
def test_unsupported_onnx_opset_is_rejected_at_request_boundary(client: httpx.Client) -> None:
    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "model_export",
                "project_id": "00000000-0000-0000-0000-000000000000",
                "format": "onnx",
                "options": {"opset_version": 19},
            },
            "worker": {
                "api": "keras",
                "backend": "tensorflow",
                "device_id": "cpu",
                "memory_policy": {"kind": "default"},
                "precision": "float32",
            },
        },
    )

    assert response.status_code == 422

###############################################################################
def test_removed_keras_export_operation_is_rejected(client: httpx.Client) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {"type": "keras_export", "project_id": project["id"]},
            "worker": execution_worker(client),
        },
    )

    assert response.status_code == 422

###############################################################################
def test_project_create_requires_current_backend_and_graph_contracts(
    client: httpx.Client,
) -> None:
    missing_backend = project_payload()
    missing_backend.pop("execution_backend")
    assert client.post("/api/projects", json=missing_backend).status_code == 422

    old_graph = project_payload()
    cast(dict[str, Any], old_graph["graph"])["schema_version"] = 4
    assert client.post("/api/projects", json=old_graph).status_code == 422

###############################################################################
def test_project_import_requires_current_bundle_format(client: httpx.Client) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    bundle = client.get(f"/api/projects/{project['id']}/export").json()
    bundle["format_version"] = 5

    assert client.post("/api/projects/import", json=bundle).status_code == 422

###############################################################################
def test_native_model_export_publishes_pt2_artifact(client: httpx.Client) -> None:
    payload = project_payload()
    payload["keras_enabled"] = False
    project = client.post("/api/projects", json=payload).json()
    response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {
                "type": "model_export",
                "project_id": project["id"],
                "format": "pytorch_exported_program",
                "filename_stem": "native-model",
            },
            "worker": native_execution_worker(client),
        },
    )
    assert response.status_code == 202, response.text

    job_id = response.json()["id"]
    job = wait_for_job(client, job_id)
    assert job["state"] == "succeeded", job
    assert job["artifact"]["format"] == "pytorch_exported_program"
    assert job["artifact"]["filename"] == "native-model.pt2"
    artifact = client.get(f"/api/execution/jobs/{job_id}/artifact")
    assert artifact.status_code == 200
    assert artifact.content[:2] == b"PK"

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
