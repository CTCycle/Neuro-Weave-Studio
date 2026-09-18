from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from common.path import CACHE_ROOT, PYTEST_BASETEMP_ROOT


###############################################################################
def pytest_configure(config: pytest.Config) -> None:
    cache_root = CACHE_ROOT.resolve()
    configured_cache_dir = Path(config.getini("cache_dir"))
    if not configured_cache_dir.is_absolute():
        configured_cache_dir = (Path(config.rootpath) / configured_cache_dir).resolve()
    try:
        configured_cache_dir.relative_to(cache_root)
    except ValueError as error:
        raise pytest.UsageError(
            f"pytest cache_dir must be beneath the canonical cache root {cache_root}"
        ) from error

    if config.option.basetemp is None:
        base_temp = PYTEST_BASETEMP_ROOT
        base_temp.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(base_temp)
    else:
        configured_basetemp = Path(config.option.basetemp).resolve()
        try:
            configured_basetemp.relative_to(cache_root)
        except ValueError as error:
            raise pytest.UsageError(
                f"pytest basetemp must be beneath the canonical cache root {cache_root}"
            ) from error

###############################################################################
@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("NEURO_WEAVE_ML_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("NEURO_WEAVE_RUNTIME_DISCOVERY_PATH", str(tmp_path / "environment-discovery.json"))
    with TestClient(create_app()) as test_client:
        yield test_client
