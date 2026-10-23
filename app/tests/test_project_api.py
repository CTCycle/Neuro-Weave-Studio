# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event

from configurations.settings import Settings
from domain.compilation_plan import CompilerObject
from domain.enums import ExecutionBackend
from domain.exports import ExportConversion, ExportFormat, ExportOptions
from domain.graph import (
    EdgeInstance,
    JaxNativeConfiguration,
    ModelConfiguration,
    ModelGraph,
    NativeCompilationConfiguration,
    NodeInstance,
    Position,
    ProjectExportBundle,
    TensorFlowNativeConfiguration,
)
from domain.jobs import JobResponse, JobSubmitRequest, ModelExportOperation, SmokeInferenceOperation
from domain.runtime import MemoryPolicy, RuntimeApi, WorkerConfiguration
from repositories.graphs import GraphRepository
from repositories.session import RepositoryContext
from services.errors import GraphPersistenceError
from services.keras_import_service import KerasImportService
from services.project_service import ProjectService
from services.runtime_controller import RuntimeController

# Consolidated from test_api_projects.py

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
            "schema_version": 6,
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
            "model_configuration": {},
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
    app_factory: Callable[..., FastAPI],
) -> None:
    database = tmp_path / "project-restart.db"

    with TestClient(app_factory(database_path=database)) as first:
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

    with TestClient(app_factory(database_path=database)) as second:
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

# Consolidated from test_project_route_errors.py

###############################################################################
def raise_create_conflict(
    _service: ProjectService,
    _payload: object,
) -> None:
    raise GraphPersistenceError("Project graph conflicts with the relational schema")

###############################################################################
def raise_duplicate_conflict(
    _service: ProjectService,
    _project_id: object,
    _payload: object,
) -> None:
    raise GraphPersistenceError("Project graph conflicts with the relational schema")

###############################################################################
def raise_keras_import_conflict(*_args: object, **_kwargs: object) -> NoReturn:
    raise GraphPersistenceError("Project graph conflicts with the relational schema")

###############################################################################
def test_project_import_maps_persistence_conflict_to_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProjectService, "create", raise_create_conflict)
    bundle = ProjectExportBundle.model_validate(
        {
            "project": {
                "name": "Imported model",
                "execution_backend": "pytorch",
                "keras_enabled": True,
                "graph": {"schema_version": 6},
            }
        }
    )

    response = client.post("/api/projects/import", json=bundle.model_dump(mode="json"))

    assert response.status_code == 409
    assert response.json()["detail"] == "Project graph conflicts with the relational schema"

###############################################################################
def test_project_duplicate_maps_persistence_conflict_to_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProjectService, "duplicate", raise_duplicate_conflict)

    response = client.post(
        f"/api/projects/{uuid4()}/duplicate",
        json={"name": "Copy"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Project graph conflicts with the relational schema"

###############################################################################
def test_keras_import_maps_persistence_conflict_to_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(KerasImportService, "import_architecture", raise_keras_import_conflict)

    response = client.post(
        "/api/imports/keras-architecture",
        content=b"test archive",
        headers={"Content-Type": "application/vnd.keras.zip"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Project graph conflicts with the relational schema"

###############################################################################
def test_unknown_api_get_does_not_fall_through_to_the_spa(client: TestClient) -> None:
    for path in ("/api/health", "/api/metrics/metric.missing"):
        response = client.get(path)

        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["detail"] == "API route not found"

# Consolidated from test_project_catalog_performance.py

###############################################################################
def _test_project_catalog_performance_project_payload(name: str, *, extra_dense: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "description": "Project catalog query-count regression test",
        "execution_backend": "pytorch",
        "keras_enabled": True,
        "graph": {
            "schema_version": 6,
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
                    "parameters": {"units": 2, "activation": "softmax", "use_bias": True},
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
                    "compilation": {
                        "optimizer": {
                            "id": "adam",
                            "parameters": {"learning_rate": 0.001},
                        },
                        "losses": {},
                        "metrics": {},
                    },
                    "mixed_precision": False,
                    "random_seed": 42,
                },
            },
            "export_settings": {
                "include_comments": True,
                "include_shape_assertions": True,
                "preferred_binary_format": "keras",
            },
            "future_training_configuration": {},
        },
    }
    if extra_dense:
        graph = cast(dict[str, object], payload["graph"])
        nodes = cast(list[dict[str, object]], graph["nodes"])
        edges = cast(list[dict[str, object]], graph["edges"])
        nodes.insert(
            2,
            {
                "id": "dense_2",
                "definition_id": "layer.dense",
                "definition_version": 1,
                "position": {"x": 450, "y": 0},
                "parameters": {"units": 2, "activation": "relu", "use_bias": True},
            },
        )
        edges[1] = {
            "id": "e2",
            "source_node_id": "dense",
            "source_port_id": "output",
            "target_node_id": "dense_2",
            "target_port_id": "input",
        }
        edges.append(
            {
                "id": "e3",
                "source_node_id": "dense_2",
                "source_port_id": "output",
                "target_node_id": "output",
                "target_port_id": "input",
            }
        )
    return payload

###############################################################################
def count_project_list_queries(client: TestClient) -> int:
    application = cast(FastAPI, client.app)
    repository_context = cast(RepositoryContext, application.state.repository_context)
    engine = repository_context.engine
    statements = 0

    def before_cursor_execute(*_args: object) -> None:
        nonlocal statements
        statements += 1

    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        response = client.get("/api/projects")
        assert response.status_code == 200, response.text
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)
    return statements

###############################################################################
def test_project_catalog_query_count_does_not_scale_with_project_count(client: TestClient) -> None:
    expected_shapes: dict[str, tuple[int, int]] = {}
    created = client.post(
        "/api/projects", json=_test_project_catalog_performance_project_payload("Catalog model 1", extra_dense=False)
    )
    assert created.status_code == 201, created.text
    expected_shapes[str(created.json()["id"])] = (3, 2)
    one_project_queries = count_project_list_queries(client)

    for index in range(2, 7):
        extra_dense = index % 2 == 0
        created = client.post(
            "/api/projects",
            json=_test_project_catalog_performance_project_payload(f"Catalog model {index}", extra_dense=extra_dense),
        )
        assert created.status_code == 201, created.text
        expected_shapes[str(created.json()["id"])] = (4, 3) if extra_dense else (3, 2)

    six_project_queries = count_project_list_queries(client)
    assert six_project_queries == one_project_queries
    assert six_project_queries <= 6

    listed = client.get("/api/projects")
    assert listed.status_code == 200, listed.text
    summaries = {str(item["id"]): item for item in listed.json()["projects"]}
    assert set(summaries) == set(expected_shapes)
    for project_id, (node_count, edge_count) in expected_shapes.items():
        summary = summaries[project_id]
        assert summary["node_count"] == node_count
        assert summary["edge_count"] == edge_count
        assert summary["input_count"] == 1
        assert summary["output_count"] == 1

# Consolidated from test_model_builder_export_regressions.py

###############################################################################
def _tensorflow_native(**payload: object) -> TensorFlowNativeConfiguration:
    return TensorFlowNativeConfiguration.model_validate(payload)


###############################################################################
def _jax_native(**payload: object) -> JaxNativeConfiguration:
    return JaxNativeConfiguration.model_validate(payload)

###############################################################################
def _node(node_id: str, definition_id: str, parameters: dict[str, object]) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        position=Position(x=0, y=0),
        parameters=parameters,
    )

###############################################################################
def _edge(edge_id: str, source: str, target: str) -> EdgeInstance:
    return EdgeInstance(
        id=edge_id,
        source_node_id=source,
        source_port_id="output",
        target_node_id=target,
        target_port_id="input",
    )

###############################################################################
def _dense_graph(*, jit_compile: bool = False) -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            _node("dense", "layer.dense", {"units": 3, "activation": "softmax"}),
            _node("output", "output.classification", {}),
        ],
        edges=[_edge("input-dense", "input", "dense"), _edge("dense-output", "dense", "output")],
        model_configuration=ModelConfiguration(
            native=NativeCompilationConfiguration(
                tensorflow=_tensorflow_native(
                    compiler=CompilerObject(
                        id="tensorflow_function", parameters={"jit_compile": jit_compile}
                    )
                ),
            ),
        ),
    )

###############################################################################
def _jax_transformer_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("tokens", "input.token", {"shape": [8], "dtype": "int32"}),
            _node(
                "embedding",
                "layer.embedding",
                {"vocabulary_size": 32, "embedding_dim": 4},
            ),
            _node(
                "position",
                "transformer.positional_encoding",
                {"mode": "sinusoidal"},
            ),
            _node(
                "encoder",
                "transformer.encoder_block",
                {
                    "num_heads": 2,
                    "key_dim": 2,
                    "ffn_multiplier": 2,
                    "layers": 1,
                    "activation": "gelu",
                },
            ),
            _node("output", "output.sequence", {}),
        ],
        edges=[
            _edge("tokens-embedding", "tokens", "embedding"),
            _edge("embedding-position", "embedding", "position"),
            _edge("position-encoder", "position", "encoder"),
            _edge("encoder-output", "encoder", "output"),
        ],
    )

###############################################################################
def _native_jax_multi_input_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("left", "input.tabular", {"shape": [4], "dtype": "float32"}),
            _node("right", "input.tabular", {"shape": [4], "dtype": "float32"}),
            _node("concat", "merge.concatenate", {"axis": -1}),
            _node("dense", "layer.dense", {"units": 3, "activation": "softmax"}),
            _node("output", "output.classification", {}),
        ],
        edges=[
            EdgeInstance(
                id="left-concat",
                source_node_id="left",
                source_port_id="output",
                target_node_id="concat",
                target_port_id="input_0",
            ),
            EdgeInstance(
                id="right-concat",
                source_node_id="right",
                source_port_id="output",
                target_node_id="concat",
                target_port_id="input_1",
            ),
            _edge("concat-dense", "concat", "dense"),
            _edge("dense-output", "dense", "output"),
        ],
        model_configuration=ModelConfiguration(
            native=NativeCompilationConfiguration(
                jax=_jax_native(
                    compiler=CompilerObject(id="jax_jit", parameters={"enabled": True})
                ),
            ),
        ),
    )

###############################################################################
def _worker(controller: RuntimeController, backend: ExecutionBackend, api: RuntimeApi) -> WorkerConfiguration:
    status = controller.backend_statuses.get(backend)
    if status is None or not status.available:
        pytest.skip(f"{backend.value} runtime is unavailable")
    device = next((item for item in status.devices if item.available), None)
    if device is None:
        pytest.skip(f"{backend.value} has no available device")
    api_status = next((item for item in status.runtime_apis if item.id is api), None)
    if api_status is None or not api_status.available:
        pytest.skip(f"{api.value}/{backend.value} runtime API is unavailable")
    return WorkerConfiguration(
        api=api,
        backend=backend,
        device_id=device.device_id,
        memory_policy=MemoryPolicy(),
        precision="float32",
    )

###############################################################################
def _wait(controller: RuntimeController, job_id: str) -> JobResponse:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = controller.get(job_id)
        if result.state.value in {"succeeded", "failed", "terminated"}:
            return result
        time.sleep(0.05)
    controller.stop(job_id)
    raise TimeoutError(f"ML worker job {job_id} timed out")

###############################################################################
def _export(
    controller: RuntimeController,
    graph: ModelGraph,
    backend: ExecutionBackend,
    api: RuntimeApi,
    export_format: ExportFormat,
    destination: Path,
) -> JobResponse:
    worker = _worker(controller, backend, api)
    job = controller.submit(
        JobSubmitRequest(
            operation=ModelExportOperation(
                project_id=uuid4(),
                format=export_format,
                filename_stem=destination.stem,
                options=ExportOptions(opset_version=18, dynamic_batch=True),
            ),
            worker=worker,
        ),
        graph=graph,
        artifact_destination=destination,
        artifact_filename=destination.name,
        project_revision=1,
    )
    result = _wait(controller, job.id)
    assert result.state.value == "succeeded", (
        result.error.message if result.error is not None else result
    )
    assert result.artifact is not None
    assert result.artifact.verification.structural
    assert result.artifact.verification.reload
    assert result.artifact.verification.inference
    assert destination.is_file()
    return result

###############################################################################
@pytest.fixture
def controller(tmp_path: Path):
    runtime = RuntimeController(
        Settings(
            ml_job_root=tmp_path / "jobs",
            export_root=tmp_path / "exports",
        )
    )
    runtime.probe_statuses(force_refresh=True)
    yield runtime
    runtime.shutdown()

###############################################################################
def test_native_tensorflow_onnx_export_inlines_jit_call_graph(
    controller: RuntimeController,
    tmp_path: Path,
) -> None:
    _export(
        controller,
        _dense_graph(jit_compile=True),
        ExecutionBackend.TENSORFLOW,
        RuntimeApi.NATIVE,
        ExportFormat.ONNX,
        tmp_path / "native_tensorflow_jit.onnx",
    )

###############################################################################
def test_native_smoke_reports_training_not_run(
    controller: RuntimeController,
) -> None:
    graph = _dense_graph()
    worker = _worker(controller, ExecutionBackend.TENSORFLOW, RuntimeApi.NATIVE)
    job = controller.submit(
        JobSubmitRequest(
            operation=SmokeInferenceOperation(project_id=uuid4(), graph=graph),
            worker=worker,
        ),
        graph=graph,
        project_revision=1,
    )
    result = _wait(controller, job.id)
    assert result.state.value == "succeeded", (
        result.error.message if result.error is not None else result
    )
    assert result.result is not None
    assert result.result["training"] == "not_run"

###############################################################################
def test_keras_jax_positional_encoding_onnx_export_uses_static_sequence_shape(
    controller: RuntimeController,
    tmp_path: Path,
) -> None:
    _export(
        controller,
        _jax_transformer_graph(),
        ExecutionBackend.JAX,
        RuntimeApi.KERAS,
        ExportFormat.ONNX,
        tmp_path / "keras_jax_positional.onnx",
    )

###############################################################################
def test_native_jax_onnx_export_uses_direct_jax2onnx_bridge(
    controller: RuntimeController,
    tmp_path: Path,
) -> None:
    result = _export(
        controller,
        _native_jax_multi_input_graph(),
        ExecutionBackend.JAX,
        RuntimeApi.NATIVE,
        ExportFormat.ONNX,
        tmp_path / "native_jax_direct.onnx",
    )
    assert result.artifact is not None
    assert result.artifact.conversion is ExportConversion.NATIVE

###############################################################################
def _missing_executable(_name: str) -> None:
    return None

###############################################################################
def test_pytorch_compile_capability_requires_msvc_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import models.pytorch_worker as pytorch_worker_module

    monkeypatch.setattr(pytorch_worker_module.sys, "platform", "win32")
    monkeypatch.setattr(pytorch_worker_module.shutil, "which", _missing_executable)

    assert pytorch_worker_module.PyTorchWorker().native_compilation_capabilities() == {
        "pytorch_compile": False
    }
