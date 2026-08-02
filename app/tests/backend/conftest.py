from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from neuro_weave.catalog.loader import get_node_catalog
from neuro_weave.config import get_database_settings, get_settings
from neuro_weave.persistence.session import get_repository_context


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("EMBEDDED_DATABASE", "true")
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    get_database_settings.cache_clear()
    get_settings.cache_clear()
    get_repository_context.cache_clear()
    get_node_catalog.cache_clear()
    from neuro_weave.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()
    get_database_settings.cache_clear()
    get_repository_context.cache_clear()
    get_node_catalog.cache_clear()
