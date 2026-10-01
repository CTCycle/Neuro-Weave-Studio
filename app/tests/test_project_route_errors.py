# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import NoReturn
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from domain.graph import ProjectExportBundle
from services.errors import GraphPersistenceError
from services.keras_import_service import KerasImportService
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
def raise_keras_import_conflict(*_args: object, **_kwargs: object) -> NoReturn:
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

###############################################################################
def test_keras_import_maps_persistence_conflict_to_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(KerasImportService, "import_architecture", raise_keras_import_conflict)

    response = client.post(
        "/api/imports/keras-architecture",
        content=b"test archive",
        headers={"Content-Type": "application/vnd.keras.zip"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Project graph conflicts with the relational schema"

###############################################################################
def test_unknown_api_get_does_not_fall_through_to_the_spa(client: TestClient) -> None:
    for path in ("/api/health", "/api/metrics/metric.missing"):
        response = client.get(path)

        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["detail"] == "API route not found"
