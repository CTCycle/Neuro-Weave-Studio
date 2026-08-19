import time
from typing import Any, cast

import httpx
import pytest
from sqlalchemy import select

from repositories.graphs import GraphRepository
from repositories.models import ProjectRecord, ProjectRevisionRecord

###############################################################################
def fail_create_graph_replace(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("graph write failed")

###############################################################################
def fail_update_graph_replace(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("graph replacement failed")

###############################################################################
def project_payload() -> dict[str, object]:
    return {
        "name": "API MLP",
        "description": "Persistence integration test",
        "code_target": "keras",
        "graph": {
            "schema_version": 2,
            "nodes": [
                {"id": "input", "definition_id": "input.tabular", "definition_version": 1, "position": {"x": 0, "y": 0}, "parameters": {"shape": [None, 8], "dtype": "float32"}},
                {"id": "dense", "definition_id": "layer.dense", "definition_version": 1, "position": {"x": 300, "y": 0}, "parameters": {"units": 2, "activation": "softmax", "use_bias": True}},
                {"id": "output", "definition_id": "output.classification", "definition_version": 1, "position": {"x": 600, "y": 0}, "parameters": {}},
            ],
            "edges": [
                {"id": "e1", "source_node_id": "input", "source_port_id": "output", "target_node_id": "dense", "target_port_id": "input"},
                {"id": "e2", "source_node_id": "dense", "source_port_id": "output", "target_node_id": "output", "target_port_id": "input"},
            ],
            "model_configuration": {"optimizer": "adam", "optimizer_parameters": {"learning_rate": 0.001}, "losses": {}, "metrics": {}, "mixed_precision": False, "random_seed": 42},
            "export_settings": {"include_comments": True, "include_shape_assertions": True, "preferred_binary_format": "keras"},
            "future_training_configuration": {},
        },
    }

###############################################################################
def test_project_save_reopen_validate_and_compile(client: httpx.Client) -> None:
    created = client.post("/api/projects", json=project_payload())
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["generated_code_snapshots"] == {}
    summaries = client.get("/api/projects")
    assert summaries.status_code == 200
    summary = next(item for item in summaries.json()["projects"] if item["id"] == project["id"])
    assert summary["node_count"] == 3
    assert summary["edge_count"] == 2
    assert summary["compiled_targets"] == []

    reopened = client.get(f"/api/projects/{project['id']}")
    assert reopened.status_code == 200
    assert reopened.json()["graph"]["nodes"][1]["definition_id"] == "layer.dense"

    validation = client.post("/api/validation", json={"graph": project["graph"], "code_target": "keras"})
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    keras_code = client.post("/api/compilation", json={"graph": project["graph"], "code_target": "keras"})
    torch_code = client.post("/api/compilation", json={"graph": project["graph"], "code_target": "pytorch"})
    assert "keras.Model" in keras_code.json()["code"]
    assert "class GeneratedModel" in torch_code.json()["code"]

    source = client.get(f"/api/exports/{project['id']}/source/keras")
    assert source.status_code == 200
    compiled_summary = next(
        item for item in client.get("/api/projects").json()["projects"] if item["id"] == project["id"]
    )
    assert compiled_summary["compiled_targets"] == ["keras"]

    update_payload = project_payload()
    update_payload["name"] = "API MLP updated"
    update_payload["expected_revision"] = project["current_revision"]
    updated = client.put(f"/api/projects/{project['id']}", json=update_payload)
    assert updated.status_code == 200
    assert updated.json()["current_revision"] == 2
    updated_summary = next(
        item for item in client.get("/api/projects").json()["projects"] if item["id"] == project["id"]
    )
    assert updated_summary["compiled_targets"] == []

    stale = client.put(f"/api/projects/{project['id']}", json=update_payload)
    assert stale.status_code == 409

    missing_revision = project_payload()
    assert client.put(f"/api/projects/{project['id']}", json=missing_revision).status_code == 422

###############################################################################
def test_health_and_metadata_are_typed_contracts(client: httpx.Client) -> None:
    health = client.get("/health")
    metadata = client.get("/api/metadata")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert metadata.status_code == 200
    assert metadata.json()["code_targets"] == ["keras", "tensorflow", "pytorch", "jax"]
    backend_items = metadata.json()["execution_backends"]
    assert {item["id"] for item in backend_items} == {
        "tensorflow", "pytorch", "jax"
    }
    assert all(isinstance(item["installed"], bool) for item in backend_items)
    assert all("version" in item for item in backend_items)

###############################################################################
def _execution_worker(client: httpx.Client) -> dict[str, object]:
    metadata = client.get("/api/metadata").json()
    status = next(item for item in metadata["execution_backends"] if item["id"] == "pytorch")
    device_id = (status.get("devices") or [{"device_id": "cpu"}])[0]["device_id"]
    return {
        "api": "keras",
        "backend": "pytorch",
        "device_id": device_id,
        "memory_policy": {"kind": "default"},
        "precision": "float32",
    }

###############################################################################
def test_keras_export_is_an_accepted_job_and_publishes_only_after_verification(client: httpx.Client) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    response = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "keras_export", "project_id": project["id"]}, "worker": _execution_worker(client)},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    job: dict[str, Any] | None = None
    for _ in range(240):
        job = cast(dict[str, Any], client.get(f"/api/execution/jobs/{job_id}").json())
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
def test_keras_export_rejects_incompatible_graph_before_worker_start(client: httpx.Client) -> None:
    payload = project_payload()
    graph = cast(dict[str, Any], payload["graph"])
    nodes = cast(list[dict[str, Any]], graph["nodes"])
    nodes[1]["definition_id"] = "unknown.node"
    project = client.post("/api/projects", json=payload).json()
    response = client.post(
        "/api/execution/jobs",
        json={"operation": {"type": "keras_export", "project_id": project["id"]}, "worker": _execution_worker(client)},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "model_validation_failed"

###############################################################################
def test_source_export_is_revision_specific(client: httpx.Client) -> None:
    created = client.post("/api/projects", json=project_payload()).json()

    source = client.get(f"/api/exports/{created['id']}/source/keras")
    assert source.status_code == 200
    assert source.json()["code_target"] == "keras"

    saved = client.get(f"/api/projects/{created['id']}").json()
    assert "keras" in saved["generated_code_snapshots"]

    update = project_payload()
    update["name"] = "New revision"
    update["expected_revision"] = saved["current_revision"]
    updated = client.put(f"/api/projects/{created['id']}", json=update)
    assert updated.status_code == 200
    assert updated.json()["generated_code_snapshots"] == {}

###############################################################################
def test_project_delete_removes_the_project(client: httpx.Client) -> None:
    created = client.post("/api/projects", json=project_payload()).json()

    deleted = client.delete(f"/api/projects/{created['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/projects/{created['id']}").status_code == 404

###############################################################################
def test_project_json_export_and_import(client: httpx.Client) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    bundle = client.get(f"/api/projects/{project['id']}/export")
    assert bundle.status_code == 200
    imported = client.post("/api/projects/import", json=bundle.json())
    assert imported.status_code == 201
    assert imported.json()["name"] == "API MLP"

###############################################################################
def test_create_rolls_back_when_graph_persistence_fails(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GraphRepository, "replace", fail_create_graph_replace)
    with pytest.raises(RuntimeError, match="graph write failed"):
        client.post("/api/projects", json=project_payload())

    context = cast(Any, client).app.state.repository_context
    with context.engine.connect() as connection:
        assert connection.execute(select(ProjectRecord)).first() is None
        assert connection.execute(select(ProjectRevisionRecord)).first() is None

###############################################################################
def test_update_rolls_back_metadata_and_revision_when_graph_persistence_fails(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
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
