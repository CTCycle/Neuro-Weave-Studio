from __future__ import annotations

from configurations.settings import REPOSITORY_ROOT

LAUNCHER = REPOSITORY_ROOT / "start_on_windows.ps1"
GITIGNORE = REPOSITORY_ROOT / ".gitignore"
ENV_EXAMPLE = REPOSITORY_ROOT / "settings/.env.example"

###############################################################################
def test_launcher_owns_the_interactive_startup_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    for menu_entry in (
        "'1' -Label 'Launch application'",
        "'2' -Label 'Install or repair dependencies'",
        "'3' -Label 'Initialize or recreate database'",
        "'4' -Label 'Run validation suite'",
        "'5' -Label 'Remove logs'",
        "'6' -Label 'Clear cache'",
        "'7' -Label 'Uninstall application'",
        "'8' -Label 'Exit'",
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
    assert "Execution backends to install" in source
    assert "tensorflow,pytorch,jax" in source
    assert "'Install' { Invoke-Install }" in source
    assert "[string]$Backends" in source
    assert "Get-InstalledExecutionBackends" in source
    assert "Get-NvidiaCudaInfo" in source
    assert "torch-cu126" in source
    assert "[string]$Acceleration" in source
    assert "Invoke-OptionalBackendSetup" in source
    assert "backend-installation.json" in source
    assert "--locked" in source
    assert "Read-InstallationType" in source
    assert "ALWAYS_REBUILD" in source
    assert "'CleanLogs' { Invoke-RemoveApplicationLogs }" in source
    assert "'ClearCache' { Invoke-ClearApplicationCache }" in source
    assert "'Uninstall' { Invoke-UninstallApplication }" in source
    assert "Type UNINSTALL to continue" in source
    assert "-m pyright" in source
    assert "app:app" not in source

###############################################################################
def test_environment_contract_is_canonical_and_secret_safe() -> None:
    template = ENV_EXAMPLE.read_text(encoding="utf-8")
    ignored = GITIGNORE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "DATABASE_SQLITE_PATH=" in template
    assert "DATABASE_CONNECT_TIMEOUT=30" in template
    assert "VITE_API_BASE_URL=/api" in template
    assert "ALWAYS_REBUILD=true" in template
    assert not (REPOSITORY_ROOT / ".env.example").exists()
    assert "ensure_environment.py" in launcher
    assert "EnvironmentExample" not in launcher
    assert "settings/.env" not in ignored
    assert "**/.env" in ignored
    assert ".env.example" not in ignored
    assert "Write-Host $value" not in launcher
    assert "Write-Host $env:" not in launcher

###############################################################################
def test_launcher_has_one_explicit_database_action_with_confirmed_reset() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "Invoke-DatabaseInitialization" in source
    assert "--reset" in source
    assert "Type RESET to continue" in source
    assert "Get-DescendantProcessIds" in source
    assert "Stop-Process -Id $processId -Force" in source

###############################################################################
def test_launcher_requires_the_canonical_frontend_lockfile() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "$clientLock = Join-Path $script:ClientDir 'package-lock.json'" in source
    assert "& $npm ci" in source
    assert "& $npm install" not in source
