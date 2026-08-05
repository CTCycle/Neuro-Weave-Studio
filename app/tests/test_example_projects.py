from __future__ import annotations

import ast
from pathlib import Path

import pytest
from common.catalogs.loader import get_node_catalog
from domain.enums import TargetFramework
from domain.graph import ProjectCreate
from services.compilation_service import CompilationService
from services.validation_service import ValidationService

EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    "filename",
    [
        "mlp_classifier.json",
        "convolutional_classifier.json",
        "lstm_sequence.json",
        "transformer_encoder.json",
    ],
)
def test_example_projects_validate_and_generate_parseable_keras_source(filename: str) -> None:
    project = ProjectCreate.model_validate_json((EXAMPLES_ROOT / filename).read_text())
    catalog = get_node_catalog()
    service = CompilationService(catalog, ValidationService(catalog))

    result = service.compile(project.graph, TargetFramework.KERAS)

    assert result.valid is True, result.diagnostics
    assert result.code is not None
    ast.parse(result.code)


def test_mlp_generates_parseable_pytorch_source() -> None:
    project = ProjectCreate.model_validate_json(
        (EXAMPLES_ROOT / "mlp_classifier.json").read_text()
    )
    catalog = get_node_catalog()
    service = CompilationService(catalog, ValidationService(catalog))

    result = service.compile(project.graph, TargetFramework.PYTORCH)

    assert result.valid is True, result.diagnostics
    assert result.code is not None
    ast.parse(result.code)
