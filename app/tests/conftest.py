from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common.catalogs.loader import get_node_catalog
from configurations.settings import get_database_settings, get_settings
from repositories.session import close_repository_context


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_EMBEDDED", "true")
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    get_settings.cache_clear()
    get_database_settings.cache_clear()
    close_repository_context()
    get_node_catalog.cache_clear()
    from app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()
    close_repository_context()
    get_database_settings.cache_clear()
    get_node_catalog.cache_clear()
