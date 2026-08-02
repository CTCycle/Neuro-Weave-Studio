from fastapi.testclient import TestClient


def project_payload() -> dict[str, object]:
    return {
        "name": "API MLP",
        "description": "Persistence integration test",
        "target_framework": "keras",
        "graph": {
            "schema_version": 1,
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


def test_project_save_reopen_validate_and_compile(client: TestClient) -> None:
    created = client.post("/api/projects", json=project_payload())
    assert created.status_code == 201, created.text
    project = created.json()

    reopened = client.get(f"/api/projects/{project['id']}")
    assert reopened.status_code == 200
    assert reopened.json()["graph"]["nodes"][1]["definition_id"] == "layer.dense"

    validation = client.post("/api/validation", json={"graph": project["graph"], "framework": "keras"})
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    keras_code = client.post("/api/compilation", json={"graph": project["graph"], "framework": "keras"})
    torch_code = client.post("/api/compilation", json={"graph": project["graph"], "framework": "pytorch"})
    assert "keras.Model" in keras_code.json()["code"]
    assert "class GeneratedModel" in torch_code.json()["code"]

    update_payload = project_payload()
    update_payload["name"] = "API MLP updated"
    update_payload["expected_revision"] = project["current_revision"]
    updated = client.put(f"/api/projects/{project['id']}", json=update_payload)
    assert updated.status_code == 200
    assert updated.json()["current_revision"] == 2


def test_project_json_export_and_import(client: TestClient) -> None:
    project = client.post("/api/projects", json=project_payload()).json()
    bundle = client.get(f"/api/projects/{project['id']}/export")
    assert bundle.status_code == 200
    imported = client.post("/api/projects/import", json=bundle.json())
    assert imported.status_code == 201
    assert imported.json()["name"] == "API MLP"
