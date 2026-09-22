from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from configurations.environment import ensure_environment_file


###############################################################################
def test_concurrent_first_start_creates_one_exact_environment_file(
    tmp_path: Path,
) -> None:
    template = tmp_path / ".env.example"
    environment_file = tmp_path / "settings" / ".env"
    expected = b"DATABASE_SQLITE_PATH=runtime/concurrent.db\n"
    template.write_bytes(expected)

    def ensure(_: int) -> Path:
        return ensure_environment_file(environment_file, template)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(ensure, range(8)))

    assert results == [environment_file] * 8
    assert environment_file.read_bytes() == expected

###############################################################################
def test_existing_environment_file_is_never_overwritten(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    environment_file = tmp_path / ".env"
    template.write_text(
        "DATABASE_SQLITE_PATH=runtime/default.db\n",
        encoding="utf-8",
    )
    original = b"DATABASE_SQLITE_PATH=local-only.db\n"
    environment_file.write_bytes(original)

    ensure_environment_file(environment_file, template)

    assert environment_file.read_bytes() == original
