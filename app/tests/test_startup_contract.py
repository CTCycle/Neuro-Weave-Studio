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
        "[3] Initialize or recreate database",
        "[4] Run validation suite",
        "[5] Exit",
    ):
        assert menu_entry in source
    assert "Assert-PortAvailable" in source
    assert "Invoke-HealthCheck" in source
    assert "Stop-ManagedProcesses" in source
    assert "$script:ClientDir = Join-Path $AppDir 'client'" in source
    assert "Invoke-FrontendBuild" in source
    assert "'run', 'preview'" in source
    assert "Front" + "endDir" not in source
    assert "Join-Path $RepoRoot 'frontend'" not in source
    assert "'run', 'dev'" not in source
    assert "-NoExit" not in source
    assert "Stop-PortListeners" not in source
    assert "'app:create_app', '--factory'" in source
    assert "[1] Keras only" in source
    assert "[2] Keras + TensorFlow" in source
    assert "[3] Keras + Torch" in source
    assert "[4] Keras + TensorFlow + Torch" in source
    assert "-m pyright" in source
    assert "app:app" not in source


def test_environment_contract_is_canonical_and_secret_safe() -> None:
    template = ENV_EXAMPLE.read_text(encoding="utf-8")
    ignored = GITIGNORE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "DATABASE_EMBEDDED=true" in template
    assert "DATABASE_SQLITE_PATH=" in template
    assert "DATABASE_URL=" in template
    assert "VITE_API_BASE_URL=/api" in template
    assert not (REPOSITORY_ROOT / ".env.example").exists()
    assert "ensure_environment.py" in launcher
    assert "EnvironmentExample" not in launcher
    assert "settings/.env" not in ignored
    assert "**/.env" in ignored
    assert ".env.example" not in ignored
    assert "Write-Host $value" not in launcher
    assert "Write-Host $env:" not in launcher


def test_launcher_has_one_explicit_database_action_with_confirmed_reset() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "Invoke-DatabaseInitialization" in source
    assert "--reset" in source
    assert "Type RESET to continue" in source
    assert "taskkill.exe /PID $child.Process.Id /T /F" in source


def test_launcher_requires_the_canonical_frontend_lockfile() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "$clientLock = Join-Path $script:ClientDir 'package-lock.json'" in source
    assert "& $npm ci" in source
    assert "& $npm install" not in source
