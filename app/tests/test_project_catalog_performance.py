from __future__ import annotations

from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event

from repositories.session import RepositoryContext

###############################################################################
def project_payload(name: str, *, extra_dense: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "description": "Project catalog query-count regression test",
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
        "/api/projects", json=project_payload("Catalog model 1", extra_dense=False)
    )
    assert created.status_code == 201, created.text
    expected_shapes[str(created.json()["id"])] = (3, 2)
    one_project_queries = count_project_list_queries(client)

    for index in range(2, 7):
        extra_dense = index % 2 == 0
        created = client.post(
            "/api/projects",
            json=project_payload(f"Catalog model {index}", extra_dense=extra_dense),
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
