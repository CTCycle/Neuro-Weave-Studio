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
def test_launch_checks_ports_before_dependency_and_frontend_work() -> None:
    launcher = launcher_source()
    launch_start = launcher.index("function Invoke-Launch {")
    launch_end = launcher.index("# Installation, database, and validation actions", launch_start)
    launch = launcher[launch_start:launch_end]

    assert launch.index("Invoke-LaunchPortGuardrail") < launch.index("Test-DependenciesReady")
    assert launch.index("Test-DependenciesReady") < launch.index("Ensure-FrontendBuildCurrent")

###############################################################################
def test_batched_port_guardrail_deduplicates_pids_and_rechecks_after_termination() -> None:
    launcher = launcher_source()
    snapshot = function_body(launcher, "Get-PortListenerRecords", "Get-ListenerProcessDetails")
    guardrail = function_body(launcher, "Invoke-LaunchPortGuardrail", "Invoke-HealthCheck")

    assert snapshot.count("Get-NetTCPConnection -State Listen") == 1
    assert "-LocalPort" not in snapshot
    assert "netstat -ano" in snapshot
    assert "Select-Object -ExpandProperty ProcessId -Unique" in guardrail
    assert guardrail.index("Confirm-LauncherAction") < guardrail.index("Stop-Process")
    assert guardrail.index("Stop-Process") < guardrail.rindex("Get-PortListenerRecords -Ports $ports")
    assert "remain occupied after approved termination" in guardrail
    assert "IsInputRedirected" in guardrail
    assert "No process was terminated" in guardrail

###############################################################################
def test_environment_fast_path_and_lightweight_dependency_probe() -> None:
    launcher = launcher_source()
    environment = function_body(launcher, "Ensure-EnvironmentFile", "Import-DotEnv")
    dependencies = function_body(launcher, "Test-DependenciesReady", "Get-ConfiguredEnvironmentValue")

    assert environment.index("PSIsContainer") < environment.index("& $python $script:EnvironmentBootstrap")
    assert "import importlib.metadata as metadata" in dependencies
    assert "import fastapi" not in dependencies
    assert "import uvicorn" not in dependencies
    assert "import sqlalchemy" not in dependencies
    assert "import alembic" not in dependencies
    assert "import keras" not in dependencies
    assert "metadata.version(name)" in dependencies
    assert "Test-FrontendDependencyCurrent" in dependencies

###############################################################################
def test_backend_probes_are_safe_for_windows_powershell_native_arguments() -> None:
    launcher = launcher_source()
    backend_probe = function_body(
        launcher,
        "Get-InstalledExecutionBackendDetails",
        "Get-InstalledExecutionBackends",
    )
    capability_probe = function_body(
        launcher,
        "Get-BackendCapabilityReport",
        "Get-BackendStatusSnapshot",
    )
    management = function_body(launcher, "Invoke-BackendManagement", "Invoke-OptionalBackendSetup")

    assert "[Text.Encoding]::UTF8.GetBytes($probe)" in backend_probe
    assert "exec(base64.b64decode('$encodedProbe'))" in backend_probe
    assert "foreach ($item in $parsed) { Write-Output $item }" in backend_probe
    assert "$ErrorActionPreference = 'Continue'" in capability_probe
    assert "$ErrorActionPreference = $previousErrorActionPreference" in capability_probe
    assert "-IncludeExportTools:$includeExportTools -SkipFrontend" in management

###############################################################################
def test_frontend_dependency_and_build_state_contracts_are_deterministic() -> None:
    launcher = launcher_source()
    inputs = function_body(launcher, "Get-FrontendBuildInputFiles", "Get-Sha256Text")
    sync = function_body(launcher, "Sync-Dependencies", "Invoke-FrontendBuild")
    build = function_body(launcher, "Invoke-FrontendBuild", "Invoke-FrontendRebuild")
    ensure = function_body(launcher, "Ensure-FrontendBuildCurrent", "Sync-Dependencies")
    atomic_writer = function_body(launcher, "Write-JsonFileAtomically", "Read-FrontendBuildState")
    build_state_check = function_body(launcher, "Test-FrontendBuildCurrent", "Ensure-FrontendBuildCurrent")
    relative_path = function_body(launcher, "Get-RepositoryRelativePath", "Get-Sha256Text")

    for required_input in ("angular.json", "tsconfig.json", "tsconfig.app.json", "package.json", "package-lock.json"):
        assert required_input in inputs
    assert "Join-Path $script:ClientDir 'src'" in inputs
    assert "*.test.*" in inputs and "*.spec.*" in inputs
    assert "e2e" not in inputs
    assert "playwright.config.ts" not in inputs
    assert "configurations" not in inputs
    assert "Get-FrontendDependencyFingerprint" in sync
    assert sync.index("& $npm ci") < sync.index("Write-FrontendDependencyState")
    assert build.index("FrontendDistIndex") < build.index("Write-FrontendBuildState")
    assert ensure.index("Test-FrontendBuildCurrent") < ensure.index("Invoke-FrontendBuild")
    assert "missing or stale" in ensure
    assert "::Replace" in atomic_writer and ".bak" in atomic_writer
    assert "TryParse" in build_state_check
    assert "MakeRelativeUri" in relative_path
    assert "[System.IO.Path]::GetRelativePath" not in launcher

###############################################################################
def test_launcher_keeps_explicit_build_ownership_and_never_reintroduces_always_rebuild() -> None:
    launcher = launcher_source()
    install = function_body(launcher, "Invoke-Install", "Invoke-DatabaseInitialization")
    rebuild = function_body(launcher, "Invoke-FrontendRebuild", "Ensure-PlaywrightBrowser")

    assert "Invoke-FrontendBuild" in install
    assert "Invoke-FrontendBuild" in rebuild
    assert "ALWAYS_REBUILD" not in launcher

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


###############################################################################
def test_launcher_uses_configurable_repository_resource_root() -> None:
    launcher = launcher_source()
    initialization = function_body(launcher, "Initialize-ResourcePaths", "Get-UserDataTargets")

    assert "$script:ResourceDir = Join-Path $script:RepoRoot 'data'" in launcher
    assert "[string]$ResourcesDir = ''" in launcher
    assert "$env:NEURO_WEAVE_RESOURCES_DIR = $ResourcesDir" in launcher
    assert "Resolve-ConfiguredPath -Name 'NEURO_WEAVE_RESOURCES_DIR' -Default 'data'" in initialization
    assert "$script:BackendInstallState = Join-Path $script:UserDataRuntimeDir 'backend-installation.json'" in initialization
    assert "app" + "/resources" not in launcher
