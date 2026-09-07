from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import event


###############################################################################
def project_payload(name: str) -> dict[str, object]:
    return {
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


###############################################################################
def count_project_list_queries(client: TestClient) -> int:
    engine = client.app.state.repository_context.engine
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
    created = client.post("/api/projects", json=project_payload("Catalog model 1"))
    assert created.status_code == 201, created.text
    one_project_queries = count_project_list_queries(client)

    for index in range(2, 7):
        created = client.post("/api/projects", json=project_payload(f"Catalog model {index}"))
        assert created.status_code == 201, created.text

    six_project_queries = count_project_list_queries(client)
    assert six_project_queries == one_project_queries
    assert six_project_queries <= 6
