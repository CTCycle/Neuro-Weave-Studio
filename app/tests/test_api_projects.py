from typing import Any, cast

import httpx
import pytest
from sqlalchemy import select

from domain.enums import ExecutionBackend
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
                {"id": "output", "definition_id": "output.classification", "definition_version": 1, "position": {"x": 600, "y": 0}, "parameters": {"name": "prediction"}},
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

    update_payload = project_payload()
    update_payload["name"] = "API MLP updated"
    update_payload["expected_revision"] = project["current_revision"]
    updated = client.put(f"/api/projects/{project['id']}", json=update_payload)
    assert updated.status_code == 200
    assert updated.json()["current_revision"] == 2

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
    assert {item["id"] for item in metadata.json()["execution_backends"]} == {
        "tensorflow", "pytorch", "jax"
    }


###############################################################################
def test_binary_export_requires_backend_and_reports_unavailable_backend(
    client: httpx.Client,
) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    missing_body = client.post(f"/api/exports/{project['id']}/keras-model")
    assert missing_body.status_code == 422

    invalid_backend = client.post(
        f"/api/exports/{project['id']}/keras-model", json={"execution_backend": "cuda"}
    )
    assert invalid_backend.status_code == 422

    missing_project = client.post(
        "/api/exports/00000000-0000-0000-0000-000000000000/keras-model",
        json={"execution_backend": "tensorflow"},
    )
    assert missing_project.status_code == 404

    cast(Any, client).app.state.backend_statuses = {}
    unavailable = client.post(
        f"/api/exports/{project['id']}/keras-model",
        json={"execution_backend": "tensorflow"},
    )
    assert unavailable.status_code == 409
    assert unavailable.json()["detail"]["code"] == "backend_unavailable"


###############################################################################
def test_binary_export_rejects_incompatible_graph_before_backend_check(
    client: httpx.Client,
) -> None:
    payload = project_payload()
    graph = cast(dict[str, Any], payload["graph"])
    nodes = cast(list[dict[str, Any]], graph["nodes"])
    nodes[1]["definition_id"] = "unknown.node"
    project = client.post("/api/projects", json=payload).json()

    cast(Any, client).app.state.backend_statuses = {
        ExecutionBackend.TENSORFLOW: {"available": False, "reason_code": "dependency_missing"}
    }
    response = client.post(
        f"/api/exports/{project['id']}/keras-model", json={"execution_backend": "tensorflow"}
    )
    assert response.status_code == 422


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
