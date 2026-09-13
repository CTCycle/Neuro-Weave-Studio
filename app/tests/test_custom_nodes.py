from __future__ import annotations

import json
import os
from typing import Self
from urllib.error import URLError

import httpx
import pytest


###############################################################################
class _FakeResponse:

    # -------------------------------------------------------------------------
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    # -------------------------------------------------------------------------
    def __enter__(self) -> Self:
        return self

    # -------------------------------------------------------------------------
    def __exit__(self, *_args: object) -> None:
        return None

    # -------------------------------------------------------------------------
    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


###############################################################################
def _stub_huggingface(monkeypatch: pytest.MonkeyPatch, sha: str = "a" * 40) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(
            {
                "sha": sha,
                "library_name": "transformers",
                "pipeline_tag": "feature-extraction",
            }
        )

    monkeypatch.setattr("services.huggingface_repository_service.urlopen", fake_urlopen)

###############################################################################
def test_custom_node_lifecycle_versions_and_builtin_locking(client: httpx.Client) -> None:
    builtins = client.get("/api/nodes/catalog")
    assert builtins.status_code == 200
    builtin = builtins.json()["nodes"][0]

    created = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Residual scale",
            "category": "custom math",
            "description": "Safe reusable tensor expression.",
            "spec": {
                "type": "mathematical",
                "formula": r"x + \frac{x}{2}",
                "input_symbols": ["x"],
            },
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    definition_id = definition["id"]
    assert definition_id.startswith("custom.node.")
    assert definition["version"] == 1
    assert definition["origin"] == "custom"
    assert definition["executable_template"] is True
    assert definition["spec"]["expression"] == "x + ((x)/(2))"

    catalog = client.get("/api/nodes/custom")
    assert catalog.status_code == 200
    assert any(item["id"] == definition_id for item in catalog.json()["definitions"])

    updated = client.put(
        f"/api/nodes/custom/{definition_id}",
        json={
            "display_name": "Residual scale v2",
            "category": "custom math",
            "description": "Edited mathematical template.",
            "spec": {
                "type": "mathematical",
                "formula": r"x^2",
                "input_symbols": ["x"],
            },
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["display_name"] == "Residual scale v2"
    assert updated.json()["version"] == 2
    assert updated.json()["spec"]["expression"] == "x**2"

    unsafe = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Unknown symbol",
            "category": "custom math",
            "description": "Uses a symbol outside the safe expression contract.",
            "spec": {"type": "mathematical", "formula": "x + y", "input_symbols": ["x"]},
        },
    )
    assert unsafe.status_code == 422

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
    assert client.delete(f"/api/nodes/custom/{builtin['id']}").status_code == 403
    assert client.delete(f"/api/nodes/custom/{definition_id}").status_code == 204

###############################################################################
def test_composite_nodes_validate_full_manifest_parameter_contract(client: httpx.Client) -> None:
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
    assert created.json()["executable_template"] is True

    invalid_node = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Invalid block",
            "category": "blocks",
            "description": "References a node that does not exist.",
            "spec": {
                "type": "composite",
                "components": [
                    {"definition_id": "layer.does_not_exist", "parameters": dict[str, object]()}
                ],
            },
        },
    )
    assert invalid_node.status_code == 422

    invalid_type = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Invalid units block",
            "category": "blocks",
            "description": "Uses a wrong parameter type.",
            "spec": {
                "type": "composite",
                "components": [{"definition_id": "layer.dense", "parameters": {"units": "many"}}],
            },
        },
    )
    assert invalid_type.status_code == 422

    invalid_option = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Invalid activation block",
            "category": "blocks",
            "description": "Uses an unsupported activation option.",
            "spec": {
                "type": "composite",
                "components": [
                    {"definition_id": "layer.activation", "parameters": {"activation": "not-real"}}
                ],
            },
        },
    )
    assert invalid_option.status_code == 422

###############################################################################
def test_repository_nodes_resolve_and_pin_public_hugging_face_reference(
    client: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_huggingface(monkeypatch)
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
    spec = created.json()["spec"]
    assert spec["repository_id"] == "google-bert/bert-base-uncased"
    assert spec["resolved_commit"] == "a" * 40
    assert spec["library_name"] == "transformers"
    assert created.json()["executable_template"] is False

    invalid = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Invalid repository reference",
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


###############################################################################
def test_repository_resolution_outage_is_service_unavailable(
    client: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise URLError("offline")

    monkeypatch.setattr("services.huggingface_repository_service.urlopen", unavailable)
    response = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Unavailable public model",
            "category": "repository",
            "description": "Resolution must fail clearly when the Hub is unavailable.",
            "spec": {
                "type": "repository",
                "repository_id": "openai-community/gpt2",
                "revision": "main",
                "artifact_type": "model",
            },
        },
    )
    assert response.status_code == 503


###############################################################################
def test_repository_resolution_rejects_invalid_commit_response(
    client: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_huggingface(monkeypatch, sha="z" * 40)
    response = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Invalid resolved commit",
            "category": "repository",
            "description": "Reject malformed provider metadata before persistence.",
            "spec": {
                "type": "repository",
                "repository_id": "openai-community/gpt2",
                "revision": "main",
                "artifact_type": "model",
            },
        },
    )
    assert response.status_code == 503


###############################################################################
@pytest.mark.skipif(
    os.getenv("NEURO_WEAVE_RUN_LIVE_REPOSITORY_TESTS") != "1",
    reason="Opt-in: exercises the live Hugging Face repository resolver.",
)
def test_repository_node_live_round_trip_and_missing_reference(client: httpx.Client) -> None:
    created = client.post(
        "/api/nodes/custom",
        json={
            "display_name": "Live public encoder reference",
            "category": "repository",
            "description": "Live verification of a public Hugging Face model reference.",
            "spec": {
                "type": "repository",
                "provider": "huggingface",
                "repository_id": "openai-community/gpt2",
                "revision": "main",
                "artifact_type": "model",
            },
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    spec = definition["spec"]
    resolved_commit = spec["resolved_commit"]
    assert spec["repository_id"] == "openai-community/gpt2"
    assert isinstance(resolved_commit, str)
    assert len(resolved_commit) == 40
    assert all(character in "0123456789abcdef" for character in resolved_commit)
    assert spec["library_name"]
    assert definition["executable_template"] is False

    try:
        missing = client.post(
            "/api/nodes/custom",
            json={
                "display_name": "Missing public repository reference",
                "category": "repository",
                "description": "Missing references must remain non-persisted and actionable.",
                "spec": {
                    "type": "repository",
                    "provider": "huggingface",
                    "repository_id": "openai-community/this-repository-does-not-exist-neuroweave",
                    "revision": "main",
                    "artifact_type": "model",
                },
            },
        )
        assert missing.status_code == 422, missing.text
    finally:
        assert client.delete(f"/api/nodes/custom/{definition['id']}").status_code == 204
