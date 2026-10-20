# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from common.path import (
    CACHE_ROOT,
    DATABASE_PATH,
    DEFAULT_RESOURCE_ROOT,
    FRONTEND_DIST_ROOT,
    PYTHON_BYTECODE_CACHE_ROOT,
    REPOSITORY_ROOT,
    RESOURCE_ROOT,
    RESOURCE_ROOT_ENV,
    RUNTIME_DISCOVERY_CACHE_PATH,
    RUNTIME_ROOT,
    resolve_resource_root,
)
from configurations.settings import Settings


###############################################################################
def test_repository_cache_paths_share_one_canonical_root() -> None:
    assert CACHE_ROOT == REPOSITORY_ROOT / "runtimes" / "cache"
    assert sys.pycache_prefix is not None
    assert Path(sys.pycache_prefix).resolve() == PYTHON_BYTECODE_CACHE_ROOT.resolve()
    assert FRONTEND_DIST_ROOT.parent == CACHE_ROOT
    assert RUNTIME_DISCOVERY_CACHE_PATH.parent == CACHE_ROOT


###############################################################################
def test_resource_root_defaults_to_repository_data() -> None:
    expected = REPOSITORY_ROOT / "data"

    assert expected == DEFAULT_RESOURCE_ROOT
    assert expected == RESOURCE_ROOT
    assert expected / "database.db" == DATABASE_PATH
    assert expected / "runtime" == RUNTIME_ROOT


###############################################################################
def test_resource_root_accepts_repository_relative_and_absolute_paths(tmp_path: Path) -> None:
    assert resolve_resource_root("alternate-resources") == REPOSITORY_ROOT / "alternate-resources"
    assert resolve_resource_root(str(tmp_path / "external-resources")) == (
        tmp_path / "external-resources"
    ).resolve()


###############################################################################
def test_resource_root_override_uses_inherited_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "configured-resources"
    monkeypatch.setenv(RESOURCE_ROOT_ENV, str(override))

    assert resolve_resource_root() == override.resolve()


###############################################################################
def test_runtime_discovery_defaults_to_the_canonical_cache_root() -> None:
    settings = Settings()

    assert settings.runtime_discovery_path == RUNTIME_DISCOVERY_CACHE_PATH.resolve()

###############################################################################
def test_runtime_discovery_cannot_be_redirected_to_persistent_application_data() -> None:
    persistent_path = Path("data/runtime/environment-discovery.json")

    with pytest.raises(ValidationError, match="canonical cache root"):
        Settings(runtime_discovery_path=persistent_path)
