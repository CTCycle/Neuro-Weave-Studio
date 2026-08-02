from pathlib import Path

from neuro_weave.config import Settings, get_database_settings
from neuro_weave.paths import REPOSITORY_ROOT


def test_relative_runtime_paths_are_anchored_to_repository_root() -> None:
    settings = Settings(
        export_root=Path("runtime/test-exports"),
    )

    assert settings.export_root == (REPOSITORY_ROOT / "runtime" / "test-exports").resolve()


def test_default_database_settings_use_the_runtime_sqlite_path(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDED_DATABASE", raising=False)
    monkeypatch.delenv("DATABASE_SQLITE_PATH", raising=False)
    get_database_settings.cache_clear()

    settings = get_database_settings()

    assert settings.embedded_database is True
    assert settings.sqlite_path == (REPOSITORY_ROOT / "app/resources/runtime/neuro_weave.db").resolve()
