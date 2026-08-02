from pathlib import Path

from neuro_weave.config import REPOSITORY_ROOT, Settings


def test_relative_runtime_paths_are_anchored_to_repository_root() -> None:
    settings = Settings(
        database_url="sqlite:///./runtime/test.db",
        export_root=Path("runtime/test-exports"),
    )

    assert settings.database_url == (
        f"sqlite:///{(REPOSITORY_ROOT / 'runtime' / 'test.db').resolve().as_posix()}"
    )
    assert settings.export_root == (REPOSITORY_ROOT / "runtime" / "test-exports").resolve()
