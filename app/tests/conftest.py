# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import create_app
from common.path import CACHE_ROOT, PYTEST_BASETEMP_ROOT, REPOSITORY_ROOT
from configurations.database_config import DatabaseConfig
from configurations.settings import Settings


###############################################################################
def pytest_configure(config: pytest.Config) -> None:
    cache_root = CACHE_ROOT.resolve()
    qa_root = (REPOSITORY_ROOT / "assets" / "QA").resolve()
    configured_cache_dir = Path(config.getini("cache_dir"))
    if not configured_cache_dir.is_absolute():
        configured_cache_dir = (Path(config.rootpath) / configured_cache_dir).resolve()
    if not any(configured_cache_dir.is_relative_to(root) for root in (cache_root, qa_root)):
        raise pytest.UsageError(
            f"pytest cache_dir must be beneath the canonical cache root {cache_root} or {qa_root}"
        )

    if config.option.basetemp is None:
        base_temp = PYTEST_BASETEMP_ROOT
        base_temp.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(base_temp)
    else:
        configured_basetemp = Path(config.option.basetemp).resolve()
        if not any(configured_basetemp.is_relative_to(root) for root in (cache_root, qa_root)):
            raise pytest.UsageError(
                f"pytest basetemp must be beneath the canonical cache root {cache_root} or {qa_root}"
            )

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
def app_factory(tmp_path: Path) -> Callable[..., FastAPI]:
    def build_app(
        *,
        database_path: Path | None = None,
        config_path: Path | None = None,
        settings: Settings | None = None,
    ) -> FastAPI:
        selected_database = database_path or (tmp_path / "test.db")
        selected_config = config_path or (tmp_path / "database-config.json")
        if not selected_config.exists():
            selected_config.parent.mkdir(parents=True, exist_ok=True)
            selected_config.write_text(
                DatabaseConfig(sqlite_path=selected_database).model_dump_json(indent=2),
                encoding="utf-8",
            )
        default_settings = Settings(
            log_file=tmp_path / "logs" / "backend.log",
            export_root=tmp_path / "exports",
            ml_job_root=tmp_path / "jobs",
            tensor_inspection_root=tmp_path / "inspections",
            runtime_discovery_path=CACHE_ROOT / "pytest-database-settings" / f"{tmp_path.name}-{selected_database.stem}.json",
        )
        return create_app(
            database_config_path=selected_config,
            environment_file=tmp_path / ".env",
            sanitize_legacy=False,
            initial_settings=settings or default_settings,
        )

    return build_app


###############################################################################
@pytest.fixture()
def client(app_factory: Callable[..., FastAPI]) -> Iterator[TestClient]:
    with TestClient(app_factory()) as test_client:
        yield test_client
