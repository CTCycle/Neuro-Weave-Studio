from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from common.path import REPOSITORY_ROOT


###############################################################################
def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        base_temp = REPOSITORY_ROOT / "assets" / "QA" / "pytest-tmp"
        base_temp.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(base_temp)


###############################################################################
@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("NEURO_WEAVE_ML_JOB_ROOT", str(tmp_path / "jobs"))
    with TestClient(create_app()) as test_client:
        yield test_client
