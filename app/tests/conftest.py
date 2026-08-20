from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from common.path import REPOSITORY_ROOT
from fastapi.testclient import TestClient

from app import create_app


###############################################################################
def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = str(REPOSITORY_ROOT / "app" / "tests" / "cache" / "pytest-tmp")

###############################################################################
@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    with TestClient(create_app()) as test_client:
        yield test_client
