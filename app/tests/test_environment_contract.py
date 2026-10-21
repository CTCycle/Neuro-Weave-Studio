# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.path import ENV_EXAMPLE_FILE
from configurations.environment import ensure_environment_file


###############################################################################
def test_concurrent_first_start_creates_one_exact_environment_file(
    tmp_path: Path,
) -> None:
    template = tmp_path / ".env.example"
    environment_file = tmp_path / "settings" / ".env"
    expected = b"FASTAPI_HOST=127.0.0.1\n"
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
        "FASTAPI_HOST=127.0.0.1\n",
        encoding="utf-8",
    )
    original = b"FASTAPI_HOST=127.0.0.1\n"
    environment_file.write_bytes(original)

    ensure_environment_file(environment_file, template)

    assert environment_file.read_bytes() == original

###############################################################################
def test_environment_template_contains_deployment_settings_not_database_settings() -> None:
    template = ENV_EXAMPLE_FILE.read_text(encoding="utf-8")

    assert "DATABASE_" not in template
    assert "NEURO_WEAVE_API_BASE_URL=/api" in template
    assert "NEURO_WEAVE_CORS_ORIGINS=" in template
    assert "NEURO_WEAVE_LOG_LEVEL" not in template
