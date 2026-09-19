from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from common.path import (
    CACHE_ROOT,
    FRONTEND_DIST_ROOT,
    PYTHON_BYTECODE_CACHE_ROOT,
    REPOSITORY_ROOT,
    RUNTIME_DISCOVERY_CACHE_PATH,
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
def test_runtime_discovery_defaults_to_the_canonical_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEURO_WEAVE_RUNTIME_DISCOVERY_PATH", raising=False)

    settings = Settings()

    assert settings.runtime_discovery_path == RUNTIME_DISCOVERY_CACHE_PATH.resolve()

###############################################################################
def test_runtime_discovery_cannot_be_redirected_to_persistent_application_data() -> None:
    persistent_path = Path("app/resources/runtime/environment-discovery.json")

    with pytest.raises(ValidationError, match="canonical cache root"):
        Settings(runtime_discovery_path=persistent_path)
