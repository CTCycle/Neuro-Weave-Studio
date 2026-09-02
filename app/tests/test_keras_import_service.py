from __future__ import annotations

import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from domain.enums import ExecutionBackend
from services.keras_import_service import KerasArchitectureImportError, parse_keras_architecture


###############################################################################
def keras_archive(document: dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("config.json", json.dumps(document))
        archive.writestr("metadata.json", json.dumps({"keras_version": "3"}))
    return buffer.getvalue()


###############################################################################
def functional_archive(input_config: dict[str, object]) -> bytes:
    return keras_archive(
        {
            "class_name": "Functional",
            "config": {
                "name": "classifier",
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "config": {"name": "input", "dtype": "float32", **input_config},
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


###############################################################################
def test_import_reconstructs_allowlisted_functional_architecture_without_weights() -> None:
    parsed = parse_keras_architecture(
        functional_archive({"batch_shape": [None, 4]}),
        execution_backend=ExecutionBackend.PYTORCH,
    )

    assert parsed.payload.name == "classifier"
    assert parsed.payload.execution_backend is ExecutionBackend.PYTORCH
    assert parsed.payload.keras_enabled is True
    assert [node.definition_id for node in parsed.payload.graph.nodes] == [
        "input.tensor",
        "layer.dense",
        "output.classification",
    ]
    assert len(parsed.payload.graph.edges) == 2
    assert dict(parsed.payload.graph.nodes[0].parameters)["shape"] == [4]
    assert parsed.payload.graph.nodes[0].parameters.get("shape") == [None, 4]
    assert any("weights" in warning for warning in parsed.warnings)


###############################################################################
@pytest.mark.parametrize(
    ("input_config", "expected_shape", "expected_definition"),
    [
        ({"batch_shape": [32, 4]}, [4], "input.tensor"),
        ({"input_shape": [12, 3]}, [12, 3], "input.sequence"),
        ({"batch_input_shape": [None, 28, 28, 1]}, [28, 28, 1], "input.image"),
    ],
)
def test_import_removes_only_the_runtime_batch_dimension(
    input_config: dict[str, object],
    expected_shape: list[int],
    expected_definition: str,
) -> None:
    parsed = parse_keras_architecture(
        functional_archive(input_config), execution_backend=ExecutionBackend.PYTORCH
    )

    input_node = parsed.payload.graph.nodes[0]
    assert input_node.definition_id == expected_definition
    assert dict(input_node.parameters)["shape"] == expected_shape
    assert input_node.parameters.get("shape") == [None, *expected_shape]


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
