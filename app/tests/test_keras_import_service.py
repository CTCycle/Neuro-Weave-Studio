from __future__ import annotations

import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from fastapi.testclient import TestClient

from domain.enums import ExecutionBackend
from services.keras_import_service import KerasArchitectureImportError, parse_keras_architecture


def keras_archive(document: dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("config.json", json.dumps(document))
        archive.writestr("metadata.json", json.dumps({"keras_version": "3"}))
    return buffer.getvalue()


def test_import_reconstructs_allowlisted_functional_architecture_without_weights() -> None:
    payload = keras_archive(
        {
            "class_name": "Functional",
            "config": {
                "name": "classifier",
                "layers": [
                    {"class_name": "InputLayer", "config": {"name": "input", "batch_shape": [None, 4], "dtype": "float32"}, "inbound_nodes": []},
                    {"class_name": "Dense", "config": {"name": "dense", "units": 2, "activation": "softmax", "use_bias": True}, "inbound_nodes": [[["input", 0, 0, {}]]]},
                ],
                "output_layers": [["dense", 0, 0]],
            },
        }
    )

    parsed = parse_keras_architecture(payload, execution_backend=ExecutionBackend.PYTORCH)

    assert parsed.payload.name == "classifier"
    assert parsed.payload.execution_backend is ExecutionBackend.PYTORCH
    assert parsed.payload.keras_enabled is True
    assert [node.definition_id for node in parsed.payload.graph.nodes] == ["input.tensor", "layer.dense", "output.classification"]
    assert len(parsed.payload.graph.edges) == 2
    assert any("weights" in warning for warning in parsed.warnings)


def test_import_rejects_executable_or_unknown_layers() -> None:
    payload = keras_archive(
        {
            "class_name": "Functional",
            "config": {
                "name": "unsafe",
                "layers": [{"class_name": "Lambda", "config": {"name": "lambda"}, "inbound_nodes": []}],
                "output_layers": [["lambda", 0, 0]],
            },
        }
    )

    with pytest.raises(KerasArchitectureImportError, match="allowlist"):
        parse_keras_architecture(payload, execution_backend=ExecutionBackend.PYTORCH)


def test_import_endpoint_creates_a_keras_enabled_project(client: TestClient) -> None:
    payload = keras_archive(
        {
            "class_name": "Sequential",
            "config": {
                "name": "sequential_model",
                "layers": [
                    {"class_name": "InputLayer", "config": {"name": "input", "batch_shape": [None, 3], "dtype": "float32"}, "inbound_nodes": []},
                    {"class_name": "Dense", "config": {"name": "dense", "units": 1, "activation": "linear", "use_bias": True}, "inbound_nodes": []},
                ],
            },
        }
    )

    response = client.post(
        "/api/imports/keras-architecture?execution_backend=pytorch",
        content=payload,
        headers={"Content-Type": "application/vnd.keras.zip"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["project"]["keras_enabled"] is True
    assert body["project"]["execution_backend"] == "pytorch"
    assert body["warnings"]
