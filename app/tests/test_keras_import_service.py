from __future__ import annotations

import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from domain.enums import ExecutionBackend
from services.keras_import_service import (
    KerasArchitectureImportError,
    parse_keras_architecture,
)


###############################################################################
def keras_archive(document: dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("config.json", json.dumps(document))
        archive.writestr("metadata.json", json.dumps({"keras_version": "3"}))
    return buffer.getvalue()


###############################################################################
def input_document(parameter_name: str, shape: list[object]) -> dict[str, object]:
    return {
        "class_name": "Functional",
        "config": {
            "name": "shape-test",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {
                        "name": "input",
                        parameter_name: shape,
                        "dtype": "float32",
                    },
                    "inbound_nodes": [],
                }
            ],
            "output_layers": [["input", 0, 0]],
        },
    }


###############################################################################
def test_import_reconstructs_allowlisted_functional_architecture_without_weights() -> None:
    payload = keras_archive(
        {
            "class_name": "Functional",
            "config": {
                "name": "classifier",
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "config": {
                            "name": "input",
                            "batch_shape": [None, 4],
                            "dtype": "float32",
                        },
                        "inbound_nodes": [],
                    },
                    {
                        "class_name": "Dense",
                        "config": {
                            "name": "dense",
                            "units": 2,
                            "activation": "softmax",
                            "use_bias": True,
                        },
                        "inbound_nodes": [[["input", 0, 0, {}]]],
                    },
                ],
                "output_layers": [["dense", 0, 0]],
            },
        }
    )

    parsed = parse_keras_architecture(payload, execution_backend=ExecutionBackend.PYTORCH)

    assert parsed.payload.name == "classifier"
    assert parsed.payload.execution_backend is ExecutionBackend.PYTORCH
    assert parsed.payload.keras_enabled is True
    assert [node.definition_id for node in parsed.payload.graph.nodes] == [
        "input.tensor",
        "layer.dense",
        "output.classification",
    ]
    assert len(parsed.payload.graph.edges) == 2
    assert parsed.payload.graph.nodes[0].parameters["shape"] == [None, 4]
    assert parsed.payload.graph.nodes[0].model_dump(mode="json")["parameters"]["shape"] == [4]
    assert any("weights" in warning for warning in parsed.warnings)


@pytest.mark.parametrize(
    ("parameter_name", "shape", "definition_id", "persisted_shape"),
    [
        ("input_shape", [8], "input.tensor", [8]),
        ("batch_shape", [None, 8], "input.tensor", [8]),
        ("batch_shape", [None, 8, 16], "input.sequence", [8, 16]),
        ("batch_input_shape", [4, 224, 224, 3], "input.image", [224, 224, 3]),
    ],
)
def test_import_normalizes_input_shapes_to_feature_only_persistence(
    parameter_name: str,
    shape: list[object],
    definition_id: str,
    persisted_shape: list[object],
) -> None:
    parsed = parse_keras_architecture(
        keras_archive(input_document(parameter_name, shape)),
        execution_backend=ExecutionBackend.PYTORCH,
    )
    input_node = parsed.payload.graph.nodes[0]

    assert input_node.definition_id == definition_id
    assert input_node.parameters["shape"] == [None, *persisted_shape]
    assert input_node.model_dump(mode="json")["parameters"]["shape"] == persisted_shape


def test_import_warns_when_discarding_a_fixed_runtime_batch_dimension() -> None:
    parsed = parse_keras_architecture(
        keras_archive(input_document("batch_shape", [16, 8])),
        execution_backend=ExecutionBackend.PYTORCH,
    )

    assert any("Fixed batch dimension 16" in warning for warning in parsed.warnings)


@pytest.mark.parametrize(
    ("parameter_name", "shape"),
    [("batch_shape", [None]), ("batch_input_shape", []), ("input_shape", [0])],
)
def test_import_rejects_empty_or_malformed_feature_shapes(
    parameter_name: str,
    shape: list[object],
) -> None:
    with pytest.raises(KerasArchitectureImportError, match="shape"):
        parse_keras_architecture(
            keras_archive(input_document(parameter_name, shape)),
            execution_backend=ExecutionBackend.PYTORCH,
        )


###############################################################################
def test_import_rejects_executable_or_unknown_layers() -> None:
    payload = keras_archive(
        {
            "class_name": "Functional",
            "config": {
                "name": "unsafe",
                "layers": [
                    {
                        "class_name": "Lambda",
                        "config": {"name": "lambda"},
                        "inbound_nodes": [],
                    }
                ],
                "output_layers": [["lambda", 0, 0]],
            },
        }
    )

    with pytest.raises(KerasArchitectureImportError, match="allowlist"):
        parse_keras_architecture(payload, execution_backend=ExecutionBackend.PYTORCH)
