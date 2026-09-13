from __future__ import annotations

import ast
from pathlib import Path

import pytest

from common.catalogs.loader import get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import ProjectCreate
from services.compilation_service import CompilationService
from services.validation_service import ValidationService

EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"

###############################################################################
@pytest.mark.parametrize(
    "filename",
    [
        "convolutional_classifier.json",
        "transformer_encoder.json",
    ],
)
def test_representative_example_projects_validate_and_generate_parseable_source(
    filename: str,
) -> None:
    project = ProjectCreate.model_validate_json(
        (EXAMPLES_ROOT / filename).read_text(encoding="utf-8")
    )
    catalog = get_node_catalog()
    service = CompilationService(catalog, ValidationService(catalog))

    result = service.compile(
        project.graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert result.valid is True, result.diagnostics
    assert result.code is not None
    ast.parse(result.code)
