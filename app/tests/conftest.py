# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

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
def _reset_jax_x64() -> None:
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return
    jax_config = getattr(jax_module, "config", None)
    updater = getattr(jax_config, "update", None)
    if callable(updater):
        cast(Callable[[str, bool], object], updater)("jax_enable_x64", False)

###############################################################################
@pytest.fixture(autouse=True)
def reset_jax_precision_state() -> Iterator[None]:
    """Prevent an int64 JAX source probe from leaking x64 into later tests."""

    _reset_jax_x64()
    yield
    _reset_jax_x64()

###############################################################################
@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("NEURO_WEAVE_ML_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("NEURO_WEAVE_RUNTIME_DISCOVERY_PATH", str(tmp_path / "environment-discovery.json"))
    with TestClient(create_app()) as test_client:
        yield test_client
