from __future__ import annotations

from pathlib import Path

###############################################################################
def launcher_source() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    return (repository_root / "start_on_windows.ps1").read_text(encoding="utf-8")


###############################################################################
def function_body(source: str, function_name: str, next_function: str) -> str:
    start = source.index(f"function {function_name} {{")
    end = source.index(f"function {next_function} {{", start)
    return source[start:end]

###############################################################################
def test_launcher_opens_browser_only_after_backend_readiness() -> None:
    launcher = launcher_source()
    launch_start = launcher.index("function Invoke-Launch {")
    launch_end = launcher.index("# Installation, database, and validation actions", launch_start)
    launch_body = launcher[launch_start:launch_end]

    frontend_ready = launch_body.index("Write-Ok 'Frontend is ready.'")
    backend_wait = launch_body.index("$backendHealth", frontend_ready)
    backend_healthy = launch_body.index("Write-Ok 'Backend is healthy.'")
    browser_open = launch_body.index("Open-ApplicationBrowser -Uri $uiUrl")

    assert frontend_ready < backend_wait < backend_healthy < browser_open

###############################################################################
def test_installation_orders_dependency_sync_database_initialization_and_build() -> None:
    launcher = launcher_source()
    install = function_body(launcher, "Invoke-Install", "Invoke-DatabaseInitialization")
    database = function_body(launcher, "Invoke-DatabaseInitialization", "Invoke-TestSuite")

    assert install.index("Sync-Dependencies") < install.index("Invoke-DatabaseInitialization") < install.index("Invoke-FrontendBuild")
    assert database.index("Test-DependenciesReady") < database.index("initialize_database.py") < database.index("& $venvPython $initializer")

###############################################################################
def test_dependency_sync_uses_lockfile_and_propagates_backend_and_frontend_failures() -> None:
    launcher = launcher_source()
    sync = function_body(launcher, "Sync-Dependencies", "Invoke-FrontendBuild")

    assert "if (Test-Path -LiteralPath (Join-Path $script:ServerDir 'uv.lock')) { $uvArguments += '--locked' }" in sync
    assert sync.index("& $uv @uvArguments") < sync.index("Backend dependency synchronization failed")
    assert sync.index("& $npm ci") < sync.index("Frontend dependency installation failed")

###############################################################################
def test_managed_process_shutdown_cleans_descendants_before_roots() -> None:
    launcher = launcher_source()
    process = function_body(launcher, "Start-ManagedProcess", "Get-DescendantProcessIds")
    shutdown = function_body(launcher, "Stop-ManagedProcesses", "Detach-ManagedProcesses")

    assert "@('/d', $mode, $command)" in process
    assert "-WindowStyle Hidden" in process
    descendant_stop = shutdown.index("foreach ($processId in $descendantIds)")
    root_stop = shutdown.index("Stop-Process -Id $rootProcessId", descendant_stop)
    assert descendant_stop < root_stop
    assert "taskkill.exe /PID $rootProcessId /T /F" in shutdown

###############################################################################
def test_launcher_rejects_invalid_ports_and_optional_backend_setup_can_reuse_state() -> None:
    launcher = launcher_source()
    port = function_body(launcher, "Get-ConfiguredInt", "Get-ListenerPids")
    optional_setup = function_body(launcher, "Invoke-OptionalBackendSetup", "Test-DevelopmentDependenciesInstalled")

    assert "valid TCP port between 1 and 65535" in port
    assert "Test-Path -LiteralPath $venvPython" in launcher
    assert "Backend setup is already complete; reused the recorded environment" in optional_setup

###############################################################################
def test_launcher_routes_generated_data_to_the_canonical_cache_root() -> None:
    launcher = launcher_source()
    environment = function_body(launcher, "Set-LauncherEnvironment", "Invoke-DownloadAndExtract")
    clear_cache = function_body(launcher, "Invoke-ClearApplicationCache", "Confirm-LauncherAction")
    test_suite = function_body(launcher, "Invoke-TestSuite", "Get-GitCommand")

    assert "$script:CacheDir = Join-Path $script:RuntimeDir 'cache'" in launcher
    assert "RuntimeCacheDir" not in launcher
    assert "app\\tests\\cache" not in launcher
    assert "assets\\QA" not in launcher
    assert "$env:PYTHONPYCACHEPREFIX = $script:PythonBytecodeCacheDir" in environment
    assert "$env:RUFF_CACHE_DIR = $script:RuffCacheDir" in environment
    assert "$env:PLAYWRIGHT_BROWSERS_PATH = $script:PlaywrightBrowsersDir" in environment
    assert "Remove-CanonicalCacheRoot -Path $script:CacheDir" in clear_cache
    assert "Set-LauncherEnvironment" in clear_cache
    assert "Ensure-PlaywrightBrowser" in test_suite
    assert "playwright install chromium" in launcher
