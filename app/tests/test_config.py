from pathlib import Path

from configurations.settings import REPOSITORY_ROOT, Settings


def test_relative_runtime_paths_are_anchored_to_repository_root() -> None:
    settings = Settings(
        export_root=Path("runtime/test-exports"),
        log_file=Path("runtime/backend.log"),
    )

    assert settings.export_root == (REPOSITORY_ROOT / "runtime/test-exports").resolve()
    assert settings.log_file == (REPOSITORY_ROOT / "runtime/backend.log").resolve()
