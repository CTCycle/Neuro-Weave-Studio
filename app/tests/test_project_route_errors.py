from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from domain.graph import ProjectExportBundle
from services.errors import GraphPersistenceError
from services.project_service import ProjectService


###############################################################################
def raise_create_conflict(
    _service: ProjectService,
    _payload: object,
) -> None:
    raise GraphPersistenceError("Project graph conflicts with the relational schema")


###############################################################################
def raise_duplicate_conflict(
    _service: ProjectService,
    _project_id: object,
    _payload: object,
) -> None:
    raise GraphPersistenceError("Project graph conflicts with the relational schema")


###############################################################################
def test_project_import_maps_persistence_conflict_to_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProjectService, "create", raise_create_conflict)
    bundle = ProjectExportBundle.model_validate(
        {
            "project": {
                "name": "Imported model",
                "execution_backend": "pytorch",
                "keras_enabled": True,
                "graph": {"schema_version": 5},
            }
        }
    )

    response = client.post("/api/projects/import", json=bundle.model_dump(mode="json"))

    assert response.status_code == 409
    assert response.json()["detail"] == "Project graph conflicts with the relational schema"


###############################################################################
def test_project_duplicate_maps_persistence_conflict_to_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProjectService, "duplicate", raise_duplicate_conflict)

    response = client.post(
        f"/api/projects/{uuid4()}/duplicate",
        json={"name": "Copy"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Project graph conflicts with the relational schema"
