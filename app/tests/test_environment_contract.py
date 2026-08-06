from __future__ import annotations

from pathlib import Path

from configurations.environment import ensure_environment_file

###############################################################################
def test_missing_environment_file_is_copied_from_template(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    environment_file = tmp_path / "settings" / ".env"
    expected = b"DATABASE_EMBEDDED=true\nDATABASE_SQLITE_PATH=runtime/test.db\n"
    template.write_bytes(expected)

    result = ensure_environment_file(environment_file, template)

    assert result == environment_file
    assert environment_file.read_bytes() == expected

###############################################################################
def test_existing_environment_file_is_never_overwritten(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    environment_file = tmp_path / ".env"
    template.write_text("DATABASE_EMBEDDED=true\n", encoding="utf-8")
    original = b"DATABASE_EMBEDDED=false\nDATABASE_PASSWORD=local-only\n"
    environment_file.write_bytes(original)

    ensure_environment_file(environment_file, template)

    assert environment_file.read_bytes() == original
