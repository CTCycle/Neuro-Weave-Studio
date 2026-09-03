from __future__ import annotations

import httpx


###############################################################################
def test_custom_node_lifecycle_and_builtin_locking(client: httpx.Client) -> None:
    builtins = client.get("/api/nodes/catalog")
    assert builtins.status_code == 200
    builtin = builtins.json()["nodes"][0]

    created = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Affine blend",
            "category": "custom math",
            "description": "Blend two tensor inputs with a scalar coefficient.",
            "spec": {
                "type": "mathematical",
                "formula": r"\\alpha x + (1-\\alpha)y",
                "input_symbols": ["x", "y"],
            },
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    definition_id = definition["id"]
    assert definition_id.startswith("custom.node.")
    assert definition["origin"] == "custom"
    assert definition["locked"] is False
    assert definition["can_edit"] is True
    assert definition["can_delete"] is True

    catalog = client.get("/api/nodes/custom")
    assert catalog.status_code == 200
    assert any(item["id"] == definition_id for item in catalog.json()["definitions"])

    updated = client.put(
        f"/api/nodes/custom/{definition_id}",
        json={
            "display_name": "Affine blend v2",
            "category": "custom math",
            "description": "Edited mathematical node definition.",
            "spec": {
                "type": "mathematical",
                "formula": r"x + \\beta y",
                "input_symbols": ["x", "y"],
            },
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["display_name"] == "Affine blend v2"

    locked_update = client.put(
        f"/api/nodes/custom/{builtin['id']}",
        json={
            "display_name": "Changed builtin",
            "category": "invalid",
            "description": "Built-ins cannot be changed.",
            "spec": {"type": "mathematical", "formula": "x", "input_symbols": ["x"]},
        },
    )
    assert locked_update.status_code == 403
    locked_delete = client.delete(f"/api/nodes/custom/{builtin['id']}")
    assert locked_delete.status_code == 403

    deleted = client.delete(f"/api/nodes/custom/{definition_id}")
    assert deleted.status_code == 204


###############################################################################
def test_composite_nodes_validate_builtin_components_and_normalize_parameters(
    client: httpx.Client,
) -> None:
    created = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Dense projection block",
            "category": "blocks",
            "description": "A reusable dense projection followed by activation.",
            "spec": {
                "type": "composite",
                "connection_mode": "sequential",
                "components": [
                    {
                        "definition_id": "layer.dense",
                        "definition_version": 1,
                        "label": "Projection",
                        "parameters": {"units": 64},
                    },
                    {
                        "definition_id": "layer.activation",
                        "definition_version": 1,
                        "label": "Activation",
                        "parameters": {"activation": "gelu"},
                    },
                ],
            },
        },
    )
    assert created.status_code == 201, created.text
    components = created.json()["spec"]["components"]
    assert components[0]["parameters"]["units"] == 64
    assert "activation" in components[0]["parameters"]

    invalid = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Invalid block",
            "category": "blocks",
            "description": "References a node that does not exist.",
            "spec": {
                "type": "composite",
                "components": [{"definition_id": "layer.does_not_exist", "parameters": {}}],
            },
        },
    )
    assert invalid.status_code == 422


###############################################################################
def test_repository_nodes_accept_safe_hugging_face_references(client: httpx.Client) -> None:
    created = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Public encoder reference",
            "category": "repository",
            "description": "Reference to a public Hugging Face model artifact.",
            "spec": {
                "type": "repository",
                "provider": "huggingface",
                "repository_id": "google-bert/bert-base-uncased",
                "revision": "main",
                "artifact_type": "model",
            },
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["spec"]["repository_id"] == "google-bert/bert-base-uncased"

    invalid = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Unsafe repository reference",
            "category": "repository",
            "description": "Invalid repository identifier.",
            "spec": {
                "type": "repository",
                "provider": "huggingface",
                "repository_id": "not-a-namespaced-id",
                "revision": "main",
                "artifact_type": "layer",
            },
        },
    )
    assert invalid.status_code == 422
