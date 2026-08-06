from __future__ import annotations

from types import SimpleNamespace
from typing import Self
from uuid import uuid4

import pytest
from pydantic import ValidationError

from domain.enums import CodeTarget
from domain.graph import ModelGraph, ProjectUpdate
from repositories.contracts import RepositoryRevisionConflictError
from services.errors import RevisionConflictError
from services.project_service import ProjectService

###############################################################################
class StubValidator:

    # -------------------------------------------------------------------------
    def validate(self, _graph: ModelGraph, _framework: CodeTarget) -> SimpleNamespace:
        return SimpleNamespace(valid=True, diagnostics=[])

###############################################################################
def test_update_requires_expected_revision_before_persistence() -> None:
    with pytest.raises(ValidationError, match="expected_revision"):
        ProjectUpdate.model_validate(
            {"name": "Project", "target_framework": CodeTarget.KERAS, "graph": ModelGraph()}
        )

###############################################################################
class ConflictProjects:

    # -------------------------------------------------------------------------
    def update(self, *_args: object, **_kwargs: object) -> None:
        raise RepositoryRevisionConflictError(1, 2)

###############################################################################
class ConflictUnitOfWork:
    projects = ConflictProjects()

    # -------------------------------------------------------------------------
    def __enter__(self) -> Self:
        return self

    # -------------------------------------------------------------------------
    def __exit__(self, *_args: object) -> None:
        return None

###############################################################################
def test_update_maps_repository_revision_conflict_to_domain_error() -> None:
    service = ProjectService(lambda: ConflictUnitOfWork(), StubValidator())  # type: ignore[arg-type]
    payload = ProjectUpdate(
        name="Project",
        code_target=CodeTarget.KERAS,
        graph=ModelGraph(),
        expected_revision=1,
    )

    with pytest.raises(RevisionConflictError, match="current 2"):
        service.update(uuid4(), payload)
