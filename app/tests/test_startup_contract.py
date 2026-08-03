from __future__ import annotations

from configurations.settings import REPOSITORY_ROOT

LAUNCHER = REPOSITORY_ROOT / "start_on_windows.ps1"
GITIGNORE = REPOSITORY_ROOT / ".gitignore"
ENV_EXAMPLE = REPOSITORY_ROOT / "settings/.env.example"


def test_launcher_owns_the_interactive_startup_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    for menu_entry in (
        "[1] Launch application",
        "[2] Install or repair dependencies",
        "[3] Initialize PostgreSQL",
        "[4] Run validation suite",
        "[5] Exit",
    ):
        assert menu_entry in source
    assert "Assert-PortAvailable" in source
    assert "Invoke-HealthCheck" in source
    assert "Stop-ManagedProcesses" in source
    assert "-NoExit" not in source
    assert "Stop-PortListeners" not in source


def test_environment_contract_is_canonical_and_secret_safe() -> None:
    template = ENV_EXAMPLE.read_text(encoding="utf-8")
    ignored = GITIGNORE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "DATABASE_EMBEDDED=true" in template
    assert "DATABASE_SQLITE_PATH=" in template
    assert "DATABASE_URL=" in template
    assert not (REPOSITORY_ROOT / ".env.example").exists()
    assert "Copy-Item -LiteralPath $script:EnvironmentExample" in launcher
    assert "Missing environment template" in launcher
    assert "settings/.env" not in ignored
    assert "**/.env" in ignored
    assert ".env.example" not in ignored
    assert "Write-Host $value" not in launcher
    assert "Write-Host $env:" not in launcher


def test_launcher_has_one_explicit_postgres_action_and_no_destructive_reset() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "Invoke-PostgresInitialization" in source
    assert "--drop-existing" not in source
    assert "drop-existing" not in source.lower()
    assert "taskkill.exe /PID $child.Process.Id /T /F" in source
