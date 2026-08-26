# ===============================================================================
# Launcher configuration
# ===============================================================================
[CmdletBinding()]
param(
    [ValidateSet('Menu', 'Install', 'RebuildFrontend', 'ManageBackends', 'Launch', 'InitDb', 'Test', 'CleanLogs', 'ClearCache', 'RemoveAllData', 'CheckForUpdates', 'Update', 'Uninstall')]
    [string]$Action = 'Menu',
    [string]$Backends = '',
    [ValidateSet('Auto', 'CPU')]
    [string]$Acceleration = 'Auto',
    [ValidateSet('Standard', 'Development')]
    [string]$InstallationType = 'Development',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest
$script:RepoRoot = $PSScriptRoot
$script:AppDir = Join-Path $RepoRoot 'app'
$script:ResourceDir = Join-Path $script:AppDir 'resources'
$script:SettingsDir = Join-Path $RepoRoot 'settings'
$script:ServerDir = Join-Path $AppDir 'server'
$script:ClientDir = Join-Path $AppDir 'client'
$script:ScriptsDir = Join-Path $AppDir 'scripts'
$script:EnvironmentBootstrap = Join-Path $ScriptsDir 'ensure_environment.py'
$script:RuntimeDir = Join-Path $RepoRoot 'runtimes'
$script:RuntimeCacheDir = Join-Path $script:RuntimeDir 'cache'
$script:CacheDir = Join-Path $AppDir 'tests\cache'
$script:UvCacheDir = Join-Path $script:RuntimeCacheDir 'uv'
$script:NpmCacheDir = Join-Path $script:RuntimeCacheDir 'npm'
$script:PipCacheDir = Join-Path $script:RuntimeCacheDir 'pip'
$script:PythonEggCacheDir = Join-Path $script:RuntimeCacheDir 'python-eggs'
$script:PythonBytecodeCacheDir = Join-Path $script:CacheDir 'python-bytecode'
$script:CoverageCacheDir = Join-Path $script:CacheDir 'coverage'
$script:PytestBaseTempDir = Join-Path $script:CacheDir 'pytest-tmp'
$script:PythonDir = Join-Path $RuntimeDir 'python'
$script:UvDir = Join-Path $RuntimeDir 'uv'
$script:NodeDir = Join-Path $RuntimeDir 'node'
$script:VenvDir = Join-Path $ServerDir '.venv'
$script:EnvironmentFile = Join-Path (Join-Path $RepoRoot 'settings') '.env'
$script:DefaultDatabasePath = Join-Path $script:ResourceDir 'database.db'
$script:UserDataRuntimeDir = Join-Path $script:ResourceDir 'runtime'
$script:LogDir = Join-Path $script:UserDataRuntimeDir 'logs'
$script:BackendInstallState = Join-Path $AppDir 'resources\runtime\backend-installation.json'
$script:FrontendDistIndex = Join-Path $script:CacheDir 'frontend-dist\index.html'
$script:GitRemote = 'origin'
$script:GitMainBranch = 'main'
$script:PythonVersion = '3.13.14'
$script:UvVersion = '0.12.0'
$script:NodeVersion = '22.23.1'
$script:Children = [System.Collections.Generic.List[object]]::new()
$script:LaunchHandedOff = $false
$script:NoBrowser = $NoBrowser

# ===============================================================================
# Output and diagnostics
# ===============================================================================
function Write-Status {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('STEP', 'OK', 'INFO', 'WARN', 'FAIL', 'RUN', 'WAIT', 'SUCCESS')]
        [string]$Level,
        [Parameter(Mandatory = $true)][string]$Message
    )

    $color = switch ($Level) {
        'STEP' { 'Cyan' }
        'OK' { 'Green' }
        'INFO' { 'DarkCyan' }
        'WARN' { 'Yellow' }
        'FAIL' { 'Red' }
        'RUN' { 'Cyan' }
        'WAIT' { 'Yellow' }
        'SUCCESS' { 'Green' }
    }
    Write-Host ("[{0}] {1}" -f $Level, $Message) -ForegroundColor $color
}

function Write-Step([string]$Message) { Write-Status -Level STEP -Message $Message }
function Write-Ok([string]$Message) { Write-Status -Level OK -Message $Message }
function Write-Info([string]$Message) { Write-Status -Level INFO -Message $Message }
function Write-Warn([string]$Message) { Write-Status -Level WARN -Message $Message }
function Write-Fatal([string]$Message) { Write-Status -Level FAIL -Message $Message }

# ===============================================================================
# Portable runtimes and environment setup
# ===============================================================================
function Get-PortablePythonPath {
    return Join-Path $script:PythonDir "cpython-$($script:PythonVersion)-windows-x86_64-none\python.exe"
}

function Get-PortablePython {
    $python = Get-PortablePythonPath
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Portable Python $($script:PythonVersion) is missing. Use the install/repair action."
    }
    return $python
}

function Assert-PythonRuntimeVersion {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    $pythonVersion = (& $PythonPath --version).Trim()
    if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne "Python $($script:PythonVersion)") {
        throw "Expected Python $($script:PythonVersion), found '$pythonVersion'."
    }
}

function Get-VenvPython {
    return Join-Path $script:VenvDir 'Scripts\python.exe'
}

function Get-Uv {
    $uv = Join-Path $script:UvDir 'uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        throw "Portable uv $($script:UvVersion) is missing. Use the install/repair action."
    }
    return $uv
}

function Get-Npm {
    $npm = Join-Path $script:NodeDir 'npm.cmd'
    if (-not (Test-Path -LiteralPath $npm)) {
        throw "Portable Node.js $($script:NodeVersion) is missing. Use the install/repair action."
    }
    return $npm
}

function Clear-PythonEnvironment {
    foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'PYTHONNOUSERSITE')) {
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
}

function Set-LauncherEnvironment {
    New-Item -ItemType Directory -Path $script:RuntimeCacheDir, $script:CacheDir, $script:UvCacheDir, `
        $script:NpmCacheDir, $script:PipCacheDir, $script:PythonEggCacheDir, $script:PythonBytecodeCacheDir, `
        $script:CoverageCacheDir, $script:PytestBaseTempDir -Force | Out-Null
    $env:UV_CACHE_DIR = $script:UvCacheDir
    $env:PIP_CACHE_DIR = $script:PipCacheDir
    $env:npm_config_cache = $script:NpmCacheDir
    $env:PYTHONPYCACHEPREFIX = $script:PythonBytecodeCacheDir
    $env:PYTHON_EGG_CACHE = $script:PythonEggCacheDir
    $env:COVERAGE_FILE = Join-Path $script:CoverageCacheDir '.coverage'
    $env:UV_PROJECT_ENVIRONMENT = $script:VenvDir
    $env:UV_LINK_MODE = 'copy'
    $nodePath = [IO.Path]::GetFullPath($script:NodeDir).TrimEnd('\')
    $pathEntries = @($env:PATH -split ';' | Where-Object {
            $_ -and [IO.Path]::GetFullPath($_).TrimEnd('\') -ine $nodePath
        })
    $env:PATH = "$nodePath;$($pathEntries -join ';')"
    Clear-PythonEnvironment
}

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    New-Item -ItemType Directory -Path (Split-Path -Parent $ArchivePath), $DestinationPath -Force | Out-Null
    Write-Info "Downloading $Uri"
    Invoke-WebRequest -Uri $Uri -OutFile $ArchivePath
    try {
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
    } finally {
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-PortableRuntimes {
    New-Item -ItemType Directory -Path $script:RuntimeDir, $script:PythonDir, $script:UvDir, $script:NodeDir -Force | Out-Null
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'The pinned NeuroWeave Node.js runtime is x64; ARM64 Windows is not supported by this launcher.'
    }

    $uv = Join-Path $script:UvDir 'uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        $archive = Join-Path $script:UvDir "uv-$($script:UvVersion).zip"
        $uri = "https://github.com/astral-sh/uv/releases/download/$($script:UvVersion)/uv-x86_64-pc-windows-msvc.zip"
        Invoke-DownloadAndExtract -Uri $uri -ArchivePath $archive -DestinationPath $script:UvDir
        $found = Get-ChildItem -LiteralPath $script:UvDir -Recurse -Filter 'uv.exe' -File | Select-Object -First 1
        if ($null -eq $found) { throw "uv.exe was not found after extracting the runtime archive." }
        if ([IO.Path]::GetFullPath($found.FullName) -ne [IO.Path]::GetFullPath($uv)) {
            Copy-Item -LiteralPath $found.FullName -Destination $uv -Force
        }
    }
    $uvVersion = (& $uv --version).Trim()
    if ($LASTEXITCODE -ne 0 -or $uvVersion -notlike "uv $($script:UvVersion)*") {
        throw "Expected uv $($script:UvVersion), found '$uvVersion'."
    }

    $python = Get-PortablePythonPath
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Step "Installing portable Python $($script:PythonVersion)"
        & $uv python install $script:PythonVersion --install-dir $script:PythonDir
        if ($LASTEXITCODE -ne 0) { throw "Portable Python installation failed with exit code $LASTEXITCODE." }
    }
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Portable Python was not found at $python after installation."
    }
    Assert-PythonRuntimeVersion -PythonPath $python

    $node = Join-Path $script:NodeDir 'node.exe'
    $npm = Join-Path $script:NodeDir 'npm.cmd'
    if (-not (Test-Path -LiteralPath $node) -or -not (Test-Path -LiteralPath $npm)) {
        $archive = Join-Path $script:NodeDir "node-v$($script:NodeVersion)-win-x64.zip"
        $uri = "https://nodejs.org/dist/v$($script:NodeVersion)/node-v$($script:NodeVersion)-win-x64.zip"
        Invoke-DownloadAndExtract -Uri $uri -ArchivePath $archive -DestinationPath $script:NodeDir
        $nested = Join-Path $script:NodeDir "node-v$($script:NodeVersion)-win-x64"
        if (Test-Path -LiteralPath $nested) {
            Get-ChildItem -LiteralPath $nested -Force | Move-Item -Destination $script:NodeDir -Force
            Remove-Item -LiteralPath $nested -Recurse -Force
        }
    }
    if (-not (Test-Path -LiteralPath $node) -or -not (Test-Path -LiteralPath $npm)) {
        throw "Node.js runtime files were not found under $($script:NodeDir)."
    }
    $nodeVersion = (& $node --version).Trim()
    if ($LASTEXITCODE -ne 0 -or $nodeVersion -ne "v$($script:NodeVersion)") {
        throw "Expected Node.js v$($script:NodeVersion), found '$nodeVersion'."
    }
    Write-Ok "Portable runtimes ready."
    Set-LauncherEnvironment
}

function Ensure-EnvironmentFile {
    $python = Get-PortablePython
    if (-not (Test-Path -LiteralPath $script:EnvironmentBootstrap)) {
        throw "Missing environment bootstrap script: $($script:EnvironmentBootstrap)"
    }
    & $python $script:EnvironmentBootstrap
    if ($LASTEXITCODE -ne 0) {
        throw "Environment initialization failed with exit code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $script:EnvironmentFile)) {
        throw "Environment bootstrap completed without creating $($script:EnvironmentFile)."
    }
}

function Import-DotEnv {
    Ensure-EnvironmentFile

    $inheritedEnvironmentKeys = @{}
    foreach ($entry in Get-ChildItem Env:) {
        $inheritedEnvironmentKeys[$entry.Name] = $true
    }
    $defaults = [ordered]@{
        FASTAPI_HOST = '127.0.0.1'
        FASTAPI_PORT = '8000'
        UI_HOST = '127.0.0.1'
        UI_PORT = '5173'
        VITE_API_BASE_URL = '/api'
        RELOAD = 'true'
        BACKEND_LOGS_VISIBLE = 'true'
    }
    foreach ($key in $defaults.Keys) {
        if (-not $inheritedEnvironmentKeys.ContainsKey($key)) {
            Set-Item -LiteralPath "Env:$key" -Value $defaults[$key]
        }
    }

    foreach ($rawLine in Get-Content -LiteralPath $script:EnvironmentFile) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#') -or $line.StartsWith(';')) { continue }
        $separator = $line.IndexOf('=')
        if ($separator -lt 1) { continue }
        $key = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim().Trim('"').Trim("'")
        if (-not $inheritedEnvironmentKeys.ContainsKey($key)) {
            Set-Item -LiteralPath "Env:$key" -Value $value
        }
    }
}

# ===============================================================================
# Compute backend management
# ===============================================================================
function Get-BooleanEnvironmentValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Default
    )

    $raw = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    if ($raw -ieq 'true') { return $true }
    if ($raw -ieq 'false') { return $false }
    throw "$Name must be either 'true' or 'false'."
}

function Get-InstalledExecutionBackendDetails {
    $venvPython = Get-VenvPython
    if (-not (Test-Path -LiteralPath $venvPython)) { return @() }
    $probe = @'
import importlib.metadata as metadata
import importlib.util
import json

frameworks = (
    {"id": "tensorflow", "package": "tensorflow", "module": "tensorflow"},
    {"id": "pytorch", "package": "torch", "module": "torch"},
    {"id": "jax", "package": "jax", "module": "jax"},
)
items = []
for framework in frameworks:
    installed = importlib.util.find_spec(framework["module"]) is not None
    version = None
    if installed:
        try:
            version = metadata.version(framework["package"])
        except metadata.PackageNotFoundError:
            pass
    items.append({**framework, "installed": installed, "version": version})
print(json.dumps(items))
'@
    $raw = (& $venvPython -c $probe 2>$null | Select-Object -Last 1)
    if ([string]::IsNullOrWhiteSpace($raw)) { return @() }
    try { return @($raw | ConvertFrom-Json) } catch { return @() }
}

function Get-InstalledExecutionBackends {
    return @(
        Get-InstalledExecutionBackendDetails |
            Where-Object { $_.installed } |
            Select-Object -ExpandProperty id
    )
}

function Get-NvidiaCudaInfo {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return [pscustomobject]@{ Available = $false; CudaVersion = $null; DeviceName = $null }
    }
    $rows = @(& $nvidiaSmi.Source --query-gpu=name,driver_version --format=csv,noheader 2>$null)
    if ($LASTEXITCODE -ne 0 -or $rows.Count -eq 0) {
        return [pscustomobject]@{ Available = $false; CudaVersion = $null; DeviceName = $null }
    }
    $columns = @([string]$rows[0] -split ',') | ForEach-Object { $_.Trim() }
    $summary = (& $nvidiaSmi.Source 2>$null) -join "`n"
    $cudaVersion = $null
    $cudaMatch = [regex]::Match($summary, 'CUDA(?: UMD)? Version:\s*(\d+(?:\.\d+)?)')
    if ($cudaMatch.Success) {
        $cudaVersion = [decimal]$cudaMatch.Groups[1].Value
    }
    return [pscustomobject]@{
        Available = $true
        CudaVersion = $cudaVersion
        DeviceName = if ($columns.Count) { $columns[0] } else { $null }
    }
}

function Get-PytorchBackendExtra {
    if ($script:Acceleration -eq 'CPU') { return 'torch-cpu' }
    $nvidia = Get-NvidiaCudaInfo
    if ($nvidia.Available -and $null -ne $nvidia.CudaVersion -and $nvidia.CudaVersion -ge [decimal]12.6) {
        return 'torch-cu126'
    }
    return 'torch-cpu'
}

function Get-BackendInstallationState {
    if (-not (Test-Path -LiteralPath $script:BackendInstallState)) { return $null }
    try {
        return Get-Content -LiteralPath $script:BackendInstallState -Raw | ConvertFrom-Json
    } catch {
        Write-Warn 'The backend installation record is unreadable; setup will be offered again.'
        return $null
    }
}

function Set-BackendInstallationState {
    param(
        [AllowNull()][AllowEmptyCollection()][string[]]$Backends = @(),
        [Parameter(Mandatory = $true)][string]$Acceleration,
        [string[]]$ResolvedExtras = @(),
        [string[]]$Failures = @(),
        [bool]$SetupCompleted = $true,
        [bool]$RestartRequired = $false
    )
    New-Item -ItemType Directory -Path (Split-Path -Parent $script:BackendInstallState) -Force | Out-Null
    $pythonVersion = $null
    $venvPython = Get-VenvPython
    if (Test-Path -LiteralPath $venvPython) { $pythonVersion = (& $venvPython --version 2>$null).Trim() }
    [ordered]@{
        schema_version = 2
        setup_completed = $SetupCompleted
        requested_backends = @($Backends)
        resolved_extras = @($ResolvedExtras)
        acceleration = $Acceleration
        python_version = $pythonVersion
        platform = "$env:OS/$env:PROCESSOR_ARCHITECTURE"
        installed = @(Get-BackendStatusSnapshot)
        failures = @($Failures)
        restart_required = $RestartRequired
        recorded_at = [DateTime]::UtcNow.ToString('O')
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $script:BackendInstallState -Encoding UTF8
}

function Clear-BackendRestartRequired {
    $state = Get-BackendInstallationState
    if ($null -eq $state -or $state.schema_version -lt 2 -or -not $state.restart_required) { return }
    $state.restart_required = $false
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $script:BackendInstallState -Encoding UTF8
}

function ConvertTo-BackendExtras {
    param([AllowEmptyString()][string]$Selection)

    if ([string]::IsNullOrWhiteSpace($Selection)) { return @() }
    $requested = @($Selection -split ',' | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
    if ($requested -contains 'auto') {
        throw "The backend selection 'auto' no longer installs a framework implicitly. Choose 'none', 'pytorch', 'tensorflow', 'jax', or 'all'."
    }
    if ($requested -contains 'all') {
        if ($requested.Count -gt 1) { throw "The backend selection 'all' cannot be combined with another backend." }
        return @('tensorflow', 'pytorch', 'jax')
    }
    if ($requested -contains 'none') {
        if ($requested.Count -gt 1) {
            throw "The backend selection 'none' cannot be combined with another backend."
        }
        return @()
    }

    $invalid = @($requested | Where-Object { $_ -notin @('tensorflow', 'pytorch', 'jax') })
    if ($invalid.Count) {
        throw "Unknown execution backend(s): $($invalid -join ', '). Keras is core and is not an execution backend. Choose tensorflow, pytorch, and/or jax."
    }
    return @($requested | Select-Object -Unique)
}

function Read-BackendSelection {
    Write-Host 'Keras 3: core model API, installed with the base application (not an execution backend).' -ForegroundColor DarkGray
    Write-Host 'Select an execution backend configuration:' -ForegroundColor White
    Write-Host '  [1] None                         - Keras core only' -ForegroundColor White
    Write-Host '  [2] PyTorch GPU / CUDA           - hardware-aware CUDA 12.6 wheel; CPU fallback if unavailable' -ForegroundColor White
    Write-Host '  [3] PyTorch CPU-only             - install the CPU wheel even when an NVIDIA GPU is detected' -ForegroundColor White
    Write-Host '  [4] TensorFlow CPU-only          - native Windows CPU execution; GPU requires WSL2 or Linux' -ForegroundColor White
    Write-Host '  [5] JAX CPU-only                 - native Windows CPU execution; Windows support is experimental' -ForegroundColor White
    Write-Host '  [6] All backends                 - PyTorch GPU / CUDA plus TensorFlow and JAX CPU' -ForegroundColor White
    Write-Host '  [7] All backends CPU-only        - CPU wheels for PyTorch, TensorFlow, and JAX' -ForegroundColor White
    $details = @(Get-InstalledExecutionBackendDetails | Where-Object { $_.installed })
    $installed = if ($details.Count) {
        @($details | ForEach-Object { "$($_.id) $($_.version)" }) -join ', '
    } else { 'none' }
    Write-Host "Currently installed execution backends: $installed" -ForegroundColor DarkCyan
    $nvidia = Get-NvidiaCudaInfo
    if ($nvidia.Available) {
        Write-Host "NVIDIA device detected: $($nvidia.DeviceName), CUDA driver $($nvidia.CudaVersion). PyTorch Auto will try the cu126 wheel." -ForegroundColor DarkGray
    } else {
        Write-Host 'No NVIDIA CUDA device detected. PyTorch Auto will use the CPU wheel.' -ForegroundColor DarkGray
    }
    $selection = (Read-Host 'Select an execution backend option (1-7)').Trim()
    switch ($selection) {
        '1' {
            $script:Acceleration = 'Auto'
            return @()
        }
        '2' {
            $script:Acceleration = 'Auto'
            return @('pytorch')
        }
        '3' {
            $script:Acceleration = 'CPU'
            return @('pytorch')
        }
        '4' {
            $script:Acceleration = 'Auto'
            return @('tensorflow')
        }
        '5' {
            $script:Acceleration = 'Auto'
            return @('jax')
        }
        '6' {
            $script:Acceleration = 'Auto'
            return @('tensorflow', 'pytorch', 'jax')
        }
        '7' {
            $script:Acceleration = 'CPU'
            return @('tensorflow', 'pytorch', 'jax')
        }
        default { throw 'Select an execution backend option from 1 to 7.' }
    }
}

function Resolve-BackendExtras {
    param([string[]]$Backends = @())

    $resolved = [System.Collections.Generic.List[string]]::new()
    foreach ($backend in @($Backends)) {
        $normalized = ([string]$backend).Trim().ToLowerInvariant()
        if (-not $normalized -or $normalized -eq 'none') { continue }
        $extra = if ($normalized -eq 'pytorch') { Get-PytorchBackendExtra } else { $normalized }
        if (-not $resolved.Contains($extra)) { $resolved.Add($extra) }
    }
    return @($resolved)
}

function Get-BackendCapabilityReport {
    $venvPython = Get-VenvPython
    if (-not (Test-Path -LiteralPath $venvPython)) { return @() }
    $backends = @(Get-InstalledExecutionBackends)
    if ($backends.Count -eq 0) { return @() }
    $reports = [System.Collections.Generic.List[object]]::new()
    Push-Location $script:ServerDir
    try {
        foreach ($backend in $backends) {
            $raw = (& $venvPython -m models.ml_worker --probe --backend $backend 2>$null | Select-Object -Last 1)
            if ($raw) {
                try { $reports.Add(($raw | ConvertFrom-Json)) } catch { }
            }
        }
    } finally { Pop-Location }
    return @($reports.ToArray())
}

function Get-BackendStatusSnapshot {
    $details = @(Get-InstalledExecutionBackendDetails)
    $detailById = @{}
    foreach ($detail in $details) { $detailById[[string]$detail.id] = $detail }
    $reports = @(Get-BackendCapabilityReport)
    $reportById = @{}
    foreach ($report in $reports) {
        $id = if ([string]$report.backend -eq 'torch') { 'pytorch' } else { [string]$report.backend }
        $reportById[$id] = $report
    }

    $snapshot = [System.Collections.Generic.List[object]]::new()
    foreach ($id in @('tensorflow', 'pytorch', 'jax')) {
        $detail = $detailById[$id]
        $report = $reportById[$id]
        $installed = $null -ne $detail -and [bool]$detail.installed
        $available = $null -ne $report -and [bool]$report.available
        $reason = if ($available) { $null } elseif ($null -ne $report -and $report.reason_code) {
            [string]$report.reason_code
        } elseif ($installed) { 'probe_failed' } else { 'dependency_missing' }
        $snapshot.Add([ordered]@{
                id = $id
                installed = $installed
                version = if ($null -ne $detail) { $detail.version } else { $null }
                available = $available
                reason_code = $reason
                accelerator = if ($available) { $report.accelerator } else { $null }
                device_name = if ($available) { $report.device_name } else { $null }
            })
    }
    return @($snapshot.ToArray())
}

function Write-BackendCapabilitySummary {
    $nvidia = Get-NvidiaCudaInfo
    if ($nvidia.Available) {
        Write-Info "NVIDIA GPU detected: $($nvidia.DeviceName) (CUDA driver $($nvidia.CudaVersion))."
    } else {
        Write-Info 'No operational NVIDIA CUDA device was detected; GPU-capable backends will use CPU fallbacks.'
    }
    Write-Info 'Native Windows limitation: TensorFlow and JAX are CPU-only; JAX CPU support is experimental.'
    foreach ($status in @(Get-BackendStatusSnapshot)) {
        $version = if ($status.version) { " v$($status.version)" } else { '' }
        if (-not $status.installed) {
            Write-Status -Level WARN -Message "$($status.id): not installed."
        } elseif ($status.available) {
            Write-Status -Level OK -Message "$($status.id)$($version): usable on $($status.accelerator) / $($status.device_name)."
        } else {
            Write-Status -Level WARN -Message "$($status.id)$($version): installed but unavailable ($($status.reason_code))."
        }
    }
}

# ===============================================================================
# Dependency and frontend lifecycle
# ===============================================================================
function Sync-Dependencies {
    param(
        [switch]$IncludeTests,
        [string[]]$BackendExtras = @(),
        [switch]$ExplicitBackendSet,
        [switch]$SkipFrontend
    )

    $uv = Get-Uv
    $python = Get-PortablePython
    Set-LauncherEnvironment
    if (-not $ExplicitBackendSet) { $BackendExtras = Get-InstalledExecutionBackends }
    $BackendExtras = @(Resolve-BackendExtras -Backends $BackendExtras)
    $uvArguments = @('sync', '--python', $python)
    if (Test-Path -LiteralPath (Join-Path $script:ServerDir 'uv.lock')) { $uvArguments += '--locked' }
    if ($IncludeTests) { $uvArguments += @('--extra', 'test') }
    foreach ($extra in $BackendExtras) { $uvArguments += @('--extra', $extra) }
    Push-Location $script:ServerDir
    try {
        Write-Step 'Synchronizing backend dependencies.'
        & $uv @uvArguments
        if ($LASTEXITCODE -ne 0) { throw "Backend dependency synchronization failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }

    if (-not $SkipFrontend) {
        $npm = Get-Npm
        $clientPackage = Join-Path $script:ClientDir 'package.json'
        $clientLock = Join-Path $script:ClientDir 'package-lock.json'
        if (-not (Test-Path -LiteralPath $clientPackage)) {
            throw "Missing frontend package manifest: $clientPackage"
        }
        if (-not (Test-Path -LiteralPath $clientLock)) {
            throw "Missing frontend lockfile: $clientLock"
        }
        Push-Location $script:ClientDir
        try {
            Write-Step 'Installing frontend dependencies.'
            & $npm ci
            if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed with exit code $LASTEXITCODE." }
        } finally { Pop-Location }
    }
    Write-Ok 'Dependencies are ready.'
}

function Invoke-FrontendBuild {
    $npm = Get-Npm
    Set-LauncherEnvironment
    Push-Location $script:ClientDir
    try {
        Write-Step 'Building frontend.'
        & $npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
}

function Invoke-FrontendRebuild {
    Ensure-PortableRuntimes
    Import-DotEnv
    Invoke-FrontendBuild
}

function Assert-BackendManagementEnvironmentStopped {
    Assert-ApplicationStopped
}

function Assert-ApplicationStopped {
    $ports = @(
        (Get-ConfiguredInt -Name 'FASTAPI_PORT' -Default '8000'),
        (Get-ConfiguredInt -Name 'UI_PORT' -Default '5173')
    ) | Select-Object -Unique
    foreach ($port in $ports) {
        $pids = @(Get-ListenerPids -Port $port)
        if ($pids.Count -gt 0) {
            throw "Cannot change the database or dependency environment while port $port is in use by PID(s) $($pids -join ', '). Close NeuroWeave and retry."
        }
    }
}

function Invoke-BackendReconciliation {
    param([AllowNull()][AllowEmptyCollection()][string[]]$RequestedBackends = @())

    $RequestedBackends = @($RequestedBackends | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $previousState = Get-BackendInstallationState
    $previousExtras = @()
    if ($null -ne $previousState -and $previousState.schema_version -ge 2 -and $null -ne $previousState.resolved_extras) {
        $previousExtras = @($previousState.resolved_extras)
    } else {
        $previousExtras = @(Resolve-BackendExtras -Backends @(Get-InstalledExecutionBackends))
    }
    $targetExtras = @(Resolve-BackendExtras -Backends $RequestedBackends)
    $failures = [System.Collections.Generic.List[string]]::new()
    $succeeded = $false
    $failureMessage = $null
    $includeTests = Test-DevelopmentDependenciesInstalled

    Write-Step "Synchronizing the requested execution environment: $(if ($RequestedBackends.Count) { $RequestedBackends -join ', ' } else { 'none (core/Keras only)' })."
    try {
        Sync-Dependencies -IncludeTests:$includeTests -BackendExtras $targetExtras -ExplicitBackendSet -SkipFrontend
        $succeeded = $true
    } catch {
        $failureMessage = $_.Exception.Message
        if ($targetExtras -contains 'torch-cu126') {
            Write-Warn 'PyTorch CUDA installation failed; retrying the complete requested set with the CPU wheel.'
            try {
                $script:Acceleration = 'CPU'
                $targetExtras = @(Resolve-BackendExtras -Backends $RequestedBackends)
                Sync-Dependencies -IncludeTests:$includeTests -BackendExtras $targetExtras -ExplicitBackendSet -SkipFrontend
                $failures.Add("PyTorch CUDA was unavailable; installed the CPU wheel instead.")
                $succeeded = $true
                Write-Ok 'PyTorch CPU fallback installed.'
            } catch {
                $failureMessage = $_.Exception.Message
            }
        }
    }

    if (-not $succeeded) {
        $restoreMessage = 'No previous backend set was available.'
        try {
            Sync-Dependencies -IncludeTests:$includeTests -BackendExtras $previousExtras -ExplicitBackendSet -SkipFrontend
            $restoreMessage = 'The previous environment was restored.'
        } catch {
            $restoreMessage = "Automatic restoration also failed: $($_.Exception.Message)"
        }
        $failures.Add($failureMessage)
        Set-BackendInstallationState -Backends $RequestedBackends -Acceleration $script:Acceleration `
            -ResolvedExtras $targetExtras -Failures @($failures.ToArray()) -SetupCompleted:$false -RestartRequired:$true
        throw "Backend update failed: $failureMessage $restoreMessage"
    }

    Set-BackendInstallationState -Backends $RequestedBackends -Acceleration $script:Acceleration `
        -ResolvedExtras $targetExtras -Failures @($failures.ToArray()) -SetupCompleted:$true -RestartRequired:$true
    Write-BackendCapabilitySummary
}

function Invoke-BackendManagement {
    Ensure-PortableRuntimes
    Import-DotEnv
    Assert-BackendManagementEnvironmentStopped
    if (-not (Test-DependenciesReady)) {
        Write-Info 'Core dependencies are missing or use another Python version; installing the base environment first.'
        Sync-Dependencies -IncludeTests:(Test-DevelopmentDependenciesInstalled) -BackendExtras @() -ExplicitBackendSet -SkipFrontend
    }
    $selection = if ([string]::IsNullOrWhiteSpace($Backends)) { @(Read-BackendSelection) } else {
        @(ConvertTo-BackendExtras -Selection $Backends)
    }
    Invoke-BackendReconciliation -RequestedBackends $selection
    Write-Ok 'Compute backend configuration saved. Relaunch NeuroWeave to use the updated environment.'
}

function Invoke-OptionalBackendSetup {
    $state = Get-BackendInstallationState
    if ([string]::IsNullOrWhiteSpace($Backends) -and $null -ne $state -and $state.schema_version -ge 2 -and $state.setup_completed -and @($state.failures).Count -eq 0) {
        Write-Ok 'Backend setup is already complete; verified the live environment instead of reinstalling packages.'
        Write-BackendCapabilitySummary
        return
    }
    if ($null -ne $state -and $state.schema_version -lt 2) {
        Write-Warn 'The backend installation record uses an older format; live backend status will be refreshed.'
    } elseif ($null -ne $state -and @($state.failures).Count -gt 0) {
        Write-Warn 'The previous backend setup was incomplete. Choose a corrected backend set or none to retry.'
    }

    $selection = if ([string]::IsNullOrWhiteSpace($Backends)) { @(Read-BackendSelection) } else {
        @(ConvertTo-BackendExtras -Selection $Backends)
    }
    Assert-BackendManagementEnvironmentStopped
    Invoke-BackendReconciliation -RequestedBackends $selection
}

function Test-DevelopmentDependenciesInstalled {
    $venvPython = Get-VenvPython
    if (-not (Test-Path -LiteralPath $venvPython)) { return $false }
    $probe = "import importlib.metadata as m; [m.version(name) for name in ('pytest', 'ruff', 'pyright')]"
    & $venvPython -c $probe *> $null
    return $LASTEXITCODE -eq 0
}

function Test-DependenciesReady {
    param([switch]$IncludeTests)

    $python = Get-PortablePythonPath
    $uv = Join-Path $script:UvDir 'uv.exe'
    $node = Join-Path $script:NodeDir 'node.exe'
    $npm = Join-Path $script:NodeDir 'npm.cmd'
    $venvPython = Get-VenvPython
    $clientPackage = Join-Path $script:ClientDir 'package.json'
    $clientLock = Join-Path $script:ClientDir 'package-lock.json'
    $clientModules = Join-Path $script:ClientDir 'node_modules'
    $clientInstallState = Join-Path $clientModules '.package-lock.json'
    $vite = Join-Path $clientModules '.bin\vite.cmd'
    if (-not (Test-Path -LiteralPath $python) -or
        -not (Test-Path -LiteralPath $uv) -or
        -not (Test-Path -LiteralPath $node) -or
        -not (Test-Path -LiteralPath $npm) -or
        -not (Test-Path -LiteralPath $venvPython) -or
        -not (Test-Path -LiteralPath (Join-Path $script:ServerDir 'app.py')) -or
        -not (Test-Path -LiteralPath $clientPackage) -or
        -not (Test-Path -LiteralPath $clientLock) -or
        -not (Test-Path -LiteralPath $clientInstallState) -or
        -not (Test-Path -LiteralPath $vite)) {
        return $false
    }
    & $python --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    try { Assert-PythonRuntimeVersion -PythonPath $python } catch { return $false }
    & $uv --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $node --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    $imports = "import fastapi, uvicorn, sqlalchemy, alembic, filelock; import importlib.metadata; importlib.metadata.version('keras')"
    if ($IncludeTests) { $imports += "; importlib.metadata.version('pytest'); importlib.metadata.version('ruff'); importlib.metadata.version('pyright')" }
    & $venvPython -c $imports *> $null
    return $LASTEXITCODE -eq 0
}

# ===============================================================================
# Application process lifecycle
# ===============================================================================
function Get-ConfiguredEnvironmentValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [AllowNull()][string]$Default = $null
    )

    $raw = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($raw) -and (Test-Path -LiteralPath $script:EnvironmentFile)) {
        foreach ($rawLine in Get-Content -LiteralPath $script:EnvironmentFile) {
            $line = $rawLine.Trim()
            if (-not $line -or $line.StartsWith('#') -or $line.StartsWith(';')) { continue }
            $separator = $line.IndexOf('=')
            if ($separator -lt 1) { continue }
            $key = $line.Substring(0, $separator).Trim()
            if ($key -ine $Name) { continue }
            $raw = $line.Substring($separator + 1).Trim().Trim('"').Trim("'")
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    return $raw
}

function Get-ConfiguredInt {
    param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][string]$Default)
    $raw = Get-ConfiguredEnvironmentValue -Name $Name -Default $Default
    $value = 0
    if (-not [int]::TryParse($raw, [System.Globalization.NumberStyles]::Integer, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$value) -or $value -lt 1 -or $value -gt 65535) {
        throw "$Name must be a valid TCP port between 1 and 65535."
    }
    return $value
}

function Get-ListenerPids {
    param([Parameter(Mandatory = $true)][int]$Port)
    try {
        $connections = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
        return @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
    } catch {
        $matches = netstat -ano | Select-String -Pattern "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$"
        return @($matches | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Sort-Object -Unique)
    }
}

function Assert-PortAvailable {
    param([Parameter(Mandatory = $true)][int]$Port)
    $pids = @(Get-ListenerPids -Port $Port)
    if ($pids.Count -gt 0) {
        throw "Port $Port is already in use by PID(s) $($pids -join ', '). Stop that service before launching NeuroWeave."
    }
}

function Invoke-HealthCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 60,
        [int]$IntervalSeconds = 1
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) { return $true }
        } catch { }
        if ([DateTime]::UtcNow -lt $deadline) { Start-Sleep -Seconds $IntervalSeconds }
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Get-BrowserHost {
    param([Parameter(Mandatory = $true)][string]$HostName)

    if ($HostName -in @('0.0.0.0', '::')) { return '127.0.0.1' }
    return $HostName
}

function Open-ApplicationBrowser {
    param([Parameter(Mandatory = $true)][string]$Uri)

    if ($script:NoBrowser) {
        Write-Info "Browser launch skipped. Open $Uri manually."
        return
    }
    try {
        Start-Process -FilePath $Uri | Out-Null
        Write-Status -Level INFO -Message "Opened $Uri in the default browser."
    } catch {
        Write-Warn "Could not open the default browser automatically. Open $Uri manually."
    }
}

function ConvertTo-CommandLineArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value -notmatch '[\s"]') { return $Value }
    return '"{0}"' -f ($Value -replace '"', '\\"')
}

function ConvertTo-CommandLine {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    $parts = @($FilePath) + @($ArgumentList) | ForEach-Object {
        ConvertTo-CommandLineArgument -Value ([string]$_)
    }
    return '"{0}"' -f ($parts -join ' ')
}

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [switch]$Visible
    )
    New-Item -ItemType Directory -Path $script:LogDir -Force | Out-Null
    $safeName = $Name.ToLowerInvariant()
    $stdout = Join-Path $script:LogDir "$safeName.out.log"
    $stderr = Join-Path $script:LogDir "$safeName.err.log"
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    $command = ConvertTo-CommandLine -FilePath $FilePath -ArgumentList $ArgumentList
    $mode = if ($Visible) { '/k' } else { '/c' }
    $startArguments = @('/d', $mode, $command)
    if ($Visible) {
        $process = Start-Process -FilePath 'cmd.exe' -ArgumentList $startArguments `
            -WorkingDirectory $WorkingDirectory -WindowStyle Normal -PassThru
        $script:Children.Add([pscustomobject]@{
                Name = $Name
                Process = $process
                Stdout = $null
                Stderr = 'the dedicated backend log terminal'
            })
    } else {
        $process = Start-Process -FilePath 'cmd.exe' -ArgumentList $startArguments `
            -WorkingDirectory $WorkingDirectory -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $script:Children.Add([pscustomobject]@{ Name = $Name; Process = $process; Stdout = $stdout; Stderr = $stderr })
    }
    return $process
}

function Get-DescendantProcessIds {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)

    try {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
        return @()
    }

    $pending = [System.Collections.Generic.Queue[int]]::new()
    $descendants = [System.Collections.Generic.List[int]]::new()
    $pending.Enqueue($RootProcessId)
    while ($pending.Count -gt 0) {
        $parentId = $pending.Dequeue()
        foreach ($process in $processes | Where-Object { [int]$_.ParentProcessId -eq $parentId }) {
            $processId = [int]$process.ProcessId
            if ($processId -ne $RootProcessId -and -not $descendants.Contains($processId)) {
                $descendants.Add($processId)
                $pending.Enqueue($processId)
            }
        }
    }
    return @($descendants)
}

function Stop-ManagedProcesses {
    for ($index = $script:Children.Count - 1; $index -ge 0; $index--) {
        $child = $script:Children[$index]
        $rootProcessId = $child.Process.Id
        $descendantIds = @(Get-DescendantProcessIds -RootProcessId $rootProcessId | Sort-Object -Descending)
        foreach ($processId in $descendantIds) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $rootProcessId -Force -ErrorAction SilentlyContinue
        if (-not $child.Process.HasExited) {
            & taskkill.exe /PID $rootProcessId /T /F *> $null
        }
    }
    $script:Children.Clear()
}

function Detach-ManagedProcesses {
    if ($script:Children.Count -gt 0) {
        Write-Info "Launcher handoff complete; detached $($script:Children.Count) application process root(s)."
    }
    $script:Children.Clear()
}

function Assert-ManagedProcessesAlive {
    foreach ($child in @($script:Children)) {
        if ($child.Process.HasExited) {
            throw "$($child.Name) exited with code $($child.Process.ExitCode). See $($child.Stderr)."
        }
    }
}

# ===============================================================================
# Application actions
# ===============================================================================
function Invoke-Launch {
    $script:LaunchHandedOff = $false
    Ensure-PortableRuntimes
    Import-DotEnv
    Set-LauncherEnvironment
    if (-not (Test-DependenciesReady)) {
        $state = Get-BackendInstallationState
        $backendExtras = if ($null -ne $state -and $state.schema_version -ge 2 -and $state.setup_completed) {
            @($state.resolved_extras)
        } else { @() }
        Sync-Dependencies -IncludeTests:(Test-DevelopmentDependenciesInstalled) -BackendExtras $backendExtras -ExplicitBackendSet
    } else {
        Write-Ok 'Application environments are ready; skipped dependency installation.'
    }
    Invoke-OptionalBackendSetup
    $backendLogsVisible = Get-BooleanEnvironmentValue -Name 'BACKEND_LOGS_VISIBLE' -Default $true
    if (-not (Test-Path -LiteralPath $script:FrontendDistIndex)) {
        Write-Info 'Frontend build output is missing after dependency recovery; rebuilding before launch.'
        Invoke-FrontendBuild
    } else {
        Write-Ok 'Frontend build output is ready; reused the existing build.'
    }

    $backendHost = [Environment]::GetEnvironmentVariable('FASTAPI_HOST', 'Process')
    $backendPort = Get-ConfiguredInt -Name 'FASTAPI_PORT' -Default '8000'
    $uiHost = [Environment]::GetEnvironmentVariable('UI_HOST', 'Process')
    $uiPort = Get-ConfiguredInt -Name 'UI_PORT' -Default '5173'
    $browserBackendHost = Get-BrowserHost -HostName $backendHost
    $browserUiHost = Get-BrowserHost -HostName $uiHost
    Assert-PortAvailable -Port $backendPort
    Assert-PortAvailable -Port $uiPort

    $venvPython = Get-VenvPython
    $npm = Get-Npm
    $backendArguments = @('-m', 'uvicorn', 'app:create_app', '--factory', '--host', $backendHost, '--port', [string]$backendPort)
    if ([Environment]::GetEnvironmentVariable('RELOAD', 'Process') -ieq 'true') { $backendArguments += '--reload' }
    try {
        $backendHealth = "http://${browserBackendHost}:$backendPort/health"
        $uiUrl = "http://${browserUiHost}:$uiPort/"
        Write-Status -Level RUN -Message "Launching backend at $backendHealth"
        $backendProcess = Start-ManagedProcess -Name 'Backend' -FilePath $venvPython -ArgumentList $backendArguments `
            -WorkingDirectory $script:ServerDir -Visible:$backendLogsVisible
        Write-Status -Level WAIT -Message "Waiting for backend readiness at $backendHealth"
        if (-not (Invoke-HealthCheck -Url $backendHealth -TimeoutSeconds 60)) {
            throw "Backend did not become healthy at $backendHealth. See $($script:LogDir)."
        }
        Write-Ok 'Backend is healthy.'

        Write-Status -Level RUN -Message "Launching frontend preview at $uiUrl"
        $frontendProcess = Start-ManagedProcess -Name 'Frontend' -FilePath $npm -ArgumentList @('run', 'preview', '--', '--host', $uiHost, '--port', [string]$uiPort, '--strictPort') -WorkingDirectory $script:ClientDir
        Write-Status -Level WAIT -Message "Waiting for frontend readiness at $uiUrl"
        if (-not (Invoke-HealthCheck -Url $uiUrl -TimeoutSeconds 60)) {
            throw "Frontend did not become ready at $uiUrl. See $($script:LogDir)."
        }
        Write-Ok 'Frontend is ready.'
        Assert-ManagedProcessesAlive
        Open-ApplicationBrowser -Uri $uiUrl
        Write-Status -Level SUCCESS -Message 'NeuroWeave Studio is running.'
        Write-Host "  Backend:  $backendHealth (PID $($backendProcess.Id))"
        Write-Host "  Frontend: $uiUrl (PID $($frontendProcess.Id))"
        Write-Host "  Logs:     $($script:LogDir)" -ForegroundColor DarkGray
        Write-Host 'The launcher will now close; the application services remain running independently.' -ForegroundColor DarkGray
        Clear-BackendRestartRequired
        Detach-ManagedProcesses
        $script:LaunchHandedOff = $true
    } finally {
        if (-not $script:LaunchHandedOff) { Stop-ManagedProcesses }
    }
}

# ===============================================================================
# Installation, database, and validation actions
# ===============================================================================
function Read-InstallationType {
    Write-Host '  [1] Development - include Ruff, Pyright, and pytest' -ForegroundColor White
    Write-Host '  [2] Standard    - install runtime dependencies only' -ForegroundColor White
    $selection = (Read-Host '  Select installation profile [1-2]').Trim()
    switch ($selection) {
        '1' { return 'Development' }
        '2' { return 'Standard' }
        default { throw 'Select an installation profile from 1 or 2.' }
    }
}

function Invoke-Install {
    param([switch]$PromptForInstallationType)

    Ensure-PortableRuntimes
    Import-DotEnv
    Assert-BackendManagementEnvironmentStopped
    $selectedInstallationType = $InstallationType
    if ($PromptForInstallationType) { $selectedInstallationType = Read-InstallationType }
    $state = Get-BackendInstallationState
    $backendSelection = if (-not [string]::IsNullOrWhiteSpace($Backends)) {
        @(ConvertTo-BackendExtras -Selection $Backends)
    } elseif ($null -ne $state -and $state.schema_version -ge 2 -and $state.setup_completed) {
        @($state.requested_backends)
    } else {
        @()
    }
    $includeTests = $selectedInstallationType -eq 'Development'
    Write-Status -Level INFO -Message "Installation profile: $selectedInstallationType"
    Sync-Dependencies -IncludeTests:$includeTests -BackendExtras $backendSelection -ExplicitBackendSet
    Invoke-DatabaseInitialization
    $resolvedExtras = @(Resolve-BackendExtras -Backends $backendSelection)
    Set-BackendInstallationState -Backends $backendSelection -Acceleration $script:Acceleration `
        -ResolvedExtras $resolvedExtras -RestartRequired:$true
    Write-BackendCapabilitySummary
    Invoke-FrontendBuild
}

function Invoke-DatabaseInitialization {
    Ensure-PortableRuntimes
    Import-DotEnv
    Assert-ApplicationStopped
    if (-not (Test-DependenciesReady)) { Sync-Dependencies }
    $venvPython = Get-VenvPython
    $initializer = Join-Path $script:ScriptsDir 'initialize_database.py'
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $script:ServerDir
        Write-Step 'Initializing or upgrading the canonical database to the Alembic head.'
        Push-Location $script:ServerDir
        try {
            & $venvPython $initializer
            if ($LASTEXITCODE -ne 0) { throw "Database initialization failed with exit code $LASTEXITCODE." }
        } finally { Pop-Location }
    } finally {
        if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $previousPythonPath }
    }
    Write-Ok 'Database initialization or upgrade completed.'
}

function Invoke-TestSuite {
    Ensure-PortableRuntimes
    Import-DotEnv
    Set-LauncherEnvironment
    if (-not (Test-DependenciesReady -IncludeTests)) { Sync-Dependencies -IncludeTests }
    $venvPython = Get-VenvPython
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $script:ServerDir
        Push-Location $script:ServerDir
        try {
            Write-Step 'Running backend Ruff, Pyright, and pytest.'
            & $venvPython -m ruff check . ..\tests ..\scripts
            if ($LASTEXITCODE -ne 0) { throw "Backend Ruff failed with exit code $LASTEXITCODE." }
            & $venvPython -m pyright
            if ($LASTEXITCODE -ne 0) { throw "Backend Pyright failed with exit code $LASTEXITCODE." }
            & $venvPython -m pytest ..\tests
            if ($LASTEXITCODE -ne 0) { throw "Backend pytest failed with exit code $LASTEXITCODE." }
        } finally { Pop-Location }
    } finally {
        if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $previousPythonPath }
    }

    $npm = Get-Npm
    Push-Location $script:ClientDir
    try {
        Write-Step 'Running frontend lint, tests, and build.'
        & $npm run lint
        if ($LASTEXITCODE -ne 0) { throw "Frontend lint failed with exit code $LASTEXITCODE." }
        & $npm run test
        if ($LASTEXITCODE -ne 0) { throw "Frontend tests failed with exit code $LASTEXITCODE." }
        & $npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
    Write-Ok 'Validation suite completed.'
}

# ===============================================================================
# Source control and update actions
# ===============================================================================
function Get-GitCommand {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $git) { throw 'Git is required for update actions but was not found on PATH.' }
    return $git.Source
}

function Get-GitText {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $git = Get-GitCommand
    $gitArguments = @('-C', $script:RepoRoot) + $Arguments
    $output = @(& $git @gitArguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        $message = @($output | ForEach-Object { [string]$_ }) -join ' '
        throw "Git command failed ($($Arguments -join ' ')): $message"
    }
    return (@($output | ForEach-Object { [string]$_ }) -join "`n").Trim()
}

function Assert-GitWorktreeClean {
    $git = Get-GitCommand
    $status = @(& $git -C $script:RepoRoot status --porcelain 2>&1)
    if ($LASTEXITCODE -ne 0) {
        $message = @($status | ForEach-Object { [string]$_ }) -join ' '
        throw "Unable to inspect the Git worktree: $message"
    }
    $changes = @($status | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    if ($changes.Count -gt 0) {
        throw "Update requires a clean Git worktree. Commit or move these changes first:`n$($changes -join "`n")"
    }
}

function Invoke-CheckForUpdates {
    $git = Get-GitCommand
    $probe = @(& $git -C $script:RepoRoot fetch --dry-run $script:GitRemote $script:GitMainBranch 2>&1)
    if ($LASTEXITCODE -ne 0) {
        $message = @($probe | ForEach-Object { [string]$_ }) -join ' '
        throw "Unable to check origin/$($script:GitMainBranch) for updates: $message"
    }

    $changes = @($probe | Where-Object {
            $line = [string]$_
            $line -match '\.\.|->\s+origin/'
        })
    if ($changes.Count -gt 0) {
        Write-Warn "A newer commit is available on origin/$($script:GitMainBranch)."
        foreach ($line in $changes) { Write-Info ([string]$line) }
    } else {
        Write-Ok "NeuroWeave is up to date with origin/$($script:GitMainBranch)."
    }
    Write-Info 'Check completed without downloading or applying changes.'
}

function Invoke-UpdateApplication {
    Assert-ApplicationStopped
    Assert-GitWorktreeClean
    $git = Get-GitCommand
    $branch = Get-GitText -Arguments @('branch', '--show-current')
    if ($branch -and $branch -ne $script:GitMainBranch) {
        Write-Warn "Current branch is '$branch'; pulling origin/$($script:GitMainBranch) into this checkout."
    }
    Write-Step "Updating the application from origin/$($script:GitMainBranch)."
    & $git -C $script:RepoRoot pull $script:GitRemote $script:GitMainBranch
    if ($LASTEXITCODE -ne 0) {
        throw "Application update failed while pulling origin/$($script:GitMainBranch)."
    }
    Write-Ok "Application update from origin/$($script:GitMainBranch) completed."
}

# ===============================================================================
# Maintenance and user data actions
# ===============================================================================
function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Child,
        [Parameter(Mandatory = $true)][string]$Parent
    )

    $childPath = [IO.Path]::GetFullPath($Child)
    $parentPath = [IO.Path]::GetFullPath($Parent)
    if ($childPath -ieq $parentPath) { return $true }
    $parentPrefix = if ($parentPath.EndsWith('\')) { $parentPath } else { "$parentPath\" }
    return $childPath.StartsWith($parentPrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Resolve-ConfiguredPath {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Default
    )

    $raw = [string](Get-ConfiguredEnvironmentValue -Name $Name -Default $Default)
    if ([string]::IsNullOrWhiteSpace($raw) -or $raw -ieq ':memory:') { return $null }
    try {
        if ([IO.Path]::IsPathRooted($raw)) {
            return [IO.Path]::GetFullPath($raw)
        }
        return [IO.Path]::GetFullPath((Join-Path $script:RepoRoot $raw))
    } catch {
        throw "$Name contains an invalid path: '$raw'."
    }
}

function Get-UserDataTargets {
    $targets = [System.Collections.Generic.List[string]]::new()
    $databasePaths = [System.Collections.Generic.List[string]]::new()
    $configuredDatabase = Resolve-ConfiguredPath -Name 'DATABASE_SQLITE_PATH' -Default 'app/resources/database.db'

    foreach ($candidate in @($script:DefaultDatabasePath, $configuredDatabase)) {
        if ($null -eq $candidate) { continue }
        $fullPath = [IO.Path]::GetFullPath($candidate)
        if ($databasePaths -notcontains $fullPath) { $databasePaths.Add($fullPath) }
    }
    if (Test-Path -LiteralPath $script:ResourceDir -PathType Container) {
        foreach ($database in @(Get-ChildItem -LiteralPath $script:ResourceDir -Filter 'database*.db' -File -Force -ErrorAction SilentlyContinue)) {
            $fullPath = [IO.Path]::GetFullPath($database.FullName)
            if ($databasePaths -notcontains $fullPath) { $databasePaths.Add($fullPath) }
        }
    }

    $targets.Add($script:UserDataRuntimeDir)
    foreach ($databasePath in $databasePaths) {
        foreach ($suffix in @('', '-wal', '-shm', '-journal')) {
            $targets.Add("$databasePath$suffix")
        }
    }

    foreach ($configuredPath in @(
            (Resolve-ConfiguredPath -Name 'NEURO_WEAVE_EXPORT_ROOT' -Default 'app/resources/runtime/exports'),
            (Resolve-ConfiguredPath -Name 'NEURO_WEAVE_LOG_FILE' -Default 'app/resources/runtime/logs/backend.log'),
            (Resolve-ConfiguredPath -Name 'NEURO_WEAVE_ML_JOB_ROOT' -Default 'app/resources/runtime/ml-jobs')
        )) {
        if ($null -ne $configuredPath -and -not (Test-PathWithin -Child $configuredPath -Parent $script:UserDataRuntimeDir)) {
            $targets.Add($configuredPath)
        }
    }
    return @($targets | Select-Object -Unique)
}

function Get-TrackedFilesUnderPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-PathWithin -Child $Path -Parent $script:RepoRoot)) { return @() }
    $relative = [IO.Path]::GetRelativePath($script:RepoRoot, [IO.Path]::GetFullPath($Path)).Replace('\', '/')
    if ([string]::IsNullOrWhiteSpace($relative) -or $relative -eq '.') {
        throw 'The repository root is not a removable user-data target.'
    }
    $git = Get-GitCommand
    $tracked = @(& $git ls-files -- $relative "$relative/**" 2>&1)
    if ($LASTEXITCODE -ne 0) {
        $message = @($tracked | ForEach-Object { [string]$_ }) -join ' '
        throw "Unable to verify whether '$Path' contains application files: $message"
    }
    return @($tracked | ForEach-Object { [string]$_ } | Where-Object { $_ })
}

function Assert-UserDataTarget {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [IO.Path]::GetFullPath($Path)
    $protectedRoots = @(
        $script:RepoRoot,
        $script:AppDir,
        $script:ResourceDir,
        $script:ServerDir,
        $script:ClientDir,
        $script:SettingsDir,
        $script:RuntimeDir,
        $script:CacheDir,
        (Join-Path $script:RepoRoot '.git')
    ) | ForEach-Object { [IO.Path]::GetFullPath($_) }
    if ([IO.Path]::GetPathRoot($fullPath) -ieq $fullPath -or $protectedRoots -contains $fullPath) {
        throw "Refusing to remove protected application path '$fullPath'."
    }

    $trackedFiles = @(Get-TrackedFilesUnderPath -Path $fullPath)
    if ($trackedFiles.Count -gt 0) {
        throw "Refusing to remove '$fullPath' because it contains tracked application files: $($trackedFiles -join ', ')"
    }
}

function Invoke-RemoveApplicationLogs {
    if (-not (Test-Path -LiteralPath $script:LogDir)) {
        Write-Info "Log directory does not exist: $($script:LogDir)"
        return
    }
    $logs = @(Get-ChildItem -LiteralPath $script:LogDir -Filter '*.log' -File -ErrorAction SilentlyContinue)
    if ($logs.Count -eq 0) {
        Write-Info 'No application log files found.'
        return
    }
    Write-Status -Level STEP -Message "Removing $($logs.Count) application log file(s)."
    $removedCount = 0
    $skippedCount = 0
    foreach ($log in $logs) {
        $result = Remove-PathBestEffort -Path $log.FullName
        $removedCount += $result.Removed
        $skippedCount += $result.Skipped
    }
    Write-Status -Level SUCCESS -Message "Removed $removedCount application log file(s); skipped $skippedCount locked or inaccessible file(s)."
}

function Invoke-RemoveAllData {
    Assert-ApplicationStopped
    $targets = @(Get-UserDataTargets)
    foreach ($target in $targets) { Assert-UserDataTarget -Path $target }

    if (-not (Confirm-LauncherAction -Prompt 'This removes all user-generated data. Type DELETE ALL DATA to continue' -Confirmation 'DELETE ALL DATA')) {
        Write-Warn 'Remove all data cancelled. Application files, settings, runtimes, dependencies, caches, and build output were preserved.'
        return
    }

    Write-Status -Level STEP -Message 'Removing user-generated application data.'
    $removedCount = 0
    $skippedCount = 0
    foreach ($target in $targets) {
        if (-not (Test-Path -LiteralPath $target -ErrorAction SilentlyContinue)) { continue }
        Write-Info "Removing $target"
        $result = Remove-PathBestEffort -Path $target
        $removedCount += $result.Removed
        $skippedCount += $result.Skipped
    }
    New-Item -ItemType Directory -Path $script:UserDataRuntimeDir -Force | Out-Null
    Write-Status -Level SUCCESS -Message "User-generated data removed ($removedCount item$(if ($removedCount -eq 1) { '' } else { 's' }) removed; $skippedCount locked or inaccessible item$(if ($skippedCount -eq 1) { '' } else { 's' }) skipped). Application files and settings were preserved."
}

function Remove-PythonCaches {
    $roots = @(
        $script:ServerDir,
        $script:ScriptsDir,
        (Join-Path $script:AppDir 'tests')
    )
    $cachePaths = @()
    foreach ($root in $roots) {
        if (Test-Path -LiteralPath $root) {
            $cachePaths += @(Get-ChildItem -LiteralPath $root -Directory -Filter '__pycache__' -Recurse -Force -ErrorAction SilentlyContinue)
        }
    }
    $cachePaths = @($cachePaths | Sort-Object FullName -Descending -Unique)
    $removedCount = 0
    $skippedCount = 0
    foreach ($cachePath in $cachePaths) {
        $result = Remove-PathBestEffort -Path $cachePath.FullName
        $removedCount += $result.Removed
        $skippedCount += $result.Skipped
    }
    return [pscustomobject]@{
        Removed = $removedCount
        Skipped = $skippedCount
    }
}

function Remove-PathBestEffort {
    param([Parameter(Mandatory = $true)][string]$Path)

    $removedCount = 0
    $skippedCount = 0
    $entryRemovalFailed = $false
    if (-not (Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue)) {
        return [pscustomobject]@{ Removed = 0; Skipped = 0 }
    }

    $entries = @(Get-ChildItem -LiteralPath $Path -Force -Recurse -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true })
    foreach ($entry in $entries) {
        try {
            Remove-Item -LiteralPath $entry.FullName -Force -ErrorAction Stop
            $removedCount++
        } catch {
            $skippedCount++
            $entryRemovalFailed = $true
            Write-Warn "Skipped removal of '$($entry.FullName)': $($_.Exception.Message)"
        }
    }

    if (-not $entryRemovalFailed -and (Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue)) {
        try {
            Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            $removedCount++
        } catch {
            $skippedCount++
            Write-Warn "Skipped removal of '$Path': $($_.Exception.Message)"
        }
    }

    return [pscustomobject]@{
        Removed = $removedCount
        Skipped = $skippedCount
    }
}

function Invoke-ClearApplicationCache {
    Write-Status -Level STEP -Message 'Removing runtime, test, and tool caches plus generated artifacts.'
    $pythonCacheResult = Remove-PythonCaches
    $legacyPaths = @(
        (Join-Path $RepoRoot 'assets\cache'),
        (Join-Path $script:RuntimeDir '.uv-cache'),
        (Join-Path $RepoRoot '.pytest_cache'),
        (Join-Path $RepoRoot '.ruff_cache'),
        (Join-Path $RepoRoot '.mypy_cache'),
        (Join-Path $script:AppDir '.pytest_cache'),
        (Join-Path $script:AppDir '.ruff_cache'),
        (Join-Path $script:AppDir 'tests\.pytest_cache'),
        (Join-Path $script:AppDir 'tests\.ruff_cache'),
        (Join-Path $script:AppDir 'tests\.mypy_cache'),
        (Join-Path $script:ServerDir '.pytest_cache'),
        (Join-Path $script:ServerDir '.ruff_cache'),
        (Join-Path $script:ServerDir '.mypy_cache'),
        (Join-Path $script:ClientDir '.vite'),
        (Join-Path $script:ClientDir 'dist'),
        (Join-Path $script:ClientDir 'coverage'),
        (Join-Path $script:ClientDir 'node_modules\.vite')
    )
    $removedCount = [int]$pythonCacheResult.Removed
    $skippedCount = [int]$pythonCacheResult.Skipped
    foreach ($target in @($script:RuntimeCacheDir, $script:CacheDir) + $legacyPaths) {
        $result = Remove-PathBestEffort -Path $target
        $removedCount += $result.Removed
        $skippedCount += $result.Skipped
    }
    New-Item -ItemType Directory -Path $script:RuntimeCacheDir, $script:CacheDir -Force | Out-Null
    Write-Status -Level SUCCESS -Message "Caches cleared ($removedCount item$(if ($removedCount -eq 1) { '' } else { 's' }) removed; $skippedCount locked or inaccessible item$(if ($skippedCount -eq 1) { '' } else { 's' }) skipped)."
}

function Confirm-LauncherAction {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [Parameter(Mandatory = $true)][string]$Confirmation
    )

    if ([Console]::IsInputRedirected) {
        throw "The destructive action requires an interactive console. Type $Confirmation when prompted."
    }
    $response = [string](Read-Host $Prompt)
    return $response.Trim() -ceq $Confirmation
}

function Invoke-UninstallApplication {
    if (-not (Confirm-LauncherAction -Prompt 'This removes runtimes, dependencies, and build output. Type UNINSTALL to continue' -Confirmation 'UNINSTALL')) {
        Write-Warn 'Uninstall cancelled. Settings, lockfiles, database files, and exports were preserved.'
        return
    }

    $targets = @(
        $script:PythonDir,
        $script:UvDir,
        $script:NodeDir,
        $script:RuntimeCacheDir,
        $script:CacheDir,
        (Join-Path $RepoRoot 'assets\cache'),
        $script:VenvDir,
        (Join-Path $script:ClientDir 'node_modules'),
        (Join-Path $script:ClientDir 'dist'),
        $script:BackendInstallState
    )
    Write-Status -Level STEP -Message 'Removing generated runtimes, dependencies, and frontend build output.'
    $removedCount = 0
    $skippedCount = 0
    foreach ($target in $targets) {
        if (-not (Test-Path -LiteralPath $target -ErrorAction SilentlyContinue)) { continue }
        Write-Info "Removing $target"
        $result = Remove-PathBestEffort -Path $target
        $removedCount += $result.Removed
        $skippedCount += $result.Skipped
    }
    $pythonCacheResult = Remove-PythonCaches
    $removedCount += $pythonCacheResult.Removed
    $skippedCount += $pythonCacheResult.Skipped
    New-Item -ItemType Directory -Path $script:RuntimeDir, $script:RuntimeCacheDir, $script:CacheDir -Force | Out-Null
    Write-Status -Level SUCCESS -Message "Application dependencies removed ($removedCount item$(if ($removedCount -eq 1) { '' } else { 's' }) removed; $skippedCount locked or inaccessible item$(if ($skippedCount -eq 1) { '' } else { 's' }) skipped). Settings, lockfiles, database files, and exports were preserved."
}

# ===============================================================================
# Non-interactive action dispatch
# ===============================================================================
if ($Action -ne 'Menu') {
    try {
        switch ($Action) {
            # Setup and validation
            'Install' { Invoke-Install }
            'RebuildFrontend' { Invoke-FrontendRebuild }
            'ManageBackends' { Invoke-BackendManagement }
            'InitDb' { Invoke-DatabaseInitialization }
            'Test' { Invoke-TestSuite }

            # Application
            'Launch' { Invoke-Launch }

            # Maintenance and data
            'CleanLogs' { Invoke-RemoveApplicationLogs }
            'ClearCache' { Invoke-ClearApplicationCache }
            'RemoveAllData' { Invoke-RemoveAllData }
            'Uninstall' { Invoke-UninstallApplication }

            # Updates
            'CheckForUpdates' { Invoke-CheckForUpdates }
            'Update' { Invoke-UpdateApplication }
        }
    } catch {
        Write-Fatal $_.Exception.Message
        exit 1
    }
    exit 0
}

function Wait-ForMenuReturn {
    if ([Console]::IsInputRedirected) { return }
    Write-Host
    Write-Host '  Press any key to return to the menu...' -ForegroundColor DarkGray
    try { [Console]::ReadKey($true) | Out-Null } catch { }
}

function Clear-MenuScreen {
    if ([Console]::IsOutputRedirected) { return }
    try { Clear-Host } catch { }
}

function Write-MenuRule {
    param([string]$Character = '-')

    Write-Host "  $($Character * 61)" -ForegroundColor DarkCyan
}

function Write-MenuOption {
    param(
        [Parameter(Mandatory = $true)][string]$Number,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Description,
        [switch]$Destructive,
        [switch]$Exit
    )

    $numberColor = if ($Destructive) { 'Yellow' } elseif ($Exit) { 'DarkGray' } else { 'Cyan' }
    $labelColor = if ($Destructive) { 'Yellow' } elseif ($Exit) { 'Gray' } else { 'White' }
    Write-Host "  [$Number] " -ForegroundColor $numberColor -NoNewline
    Write-Host $Label.PadRight(35) -ForegroundColor $labelColor -NoNewline
    Write-Host $Description -ForegroundColor DarkGray
}

function Show-LauncherMenu {
    Clear-MenuScreen
    Write-Host
    Write-MenuRule
    Write-Host '  NEUROWEAVE STUDIO' -ForegroundColor Cyan -NoNewline
    Write-Host '  /  LOCAL WORKSPACE' -ForegroundColor White
    Write-Host '  Visual machine-learning architecture editor' -ForegroundColor DarkGray
    Write-MenuRule
    Write-Host
    Write-Host '  APPLICATION' -ForegroundColor DarkCyan
    Write-MenuOption -Number '1' -Label 'Launch application' -Description 'Start local backend and frontend services'
    Write-Host
    Write-Host '  SETUP & VALIDATION' -ForegroundColor DarkCyan
    Write-MenuOption -Number '2' -Label 'Install or repair base dependencies' -Description 'Prepare runtimes and core packages'
    Write-MenuOption -Number '3' -Label 'Rebuild frontend' -Description 'Rebuild the frontend from current source'
    Write-MenuOption -Number '4' -Label 'Manage compute backends' -Description 'Install, inspect, or remove optional ML frameworks'
    Write-MenuOption -Number '5' -Label 'Initialize or upgrade database' -Description 'Create or migrate the schema to the Alembic head'
    Write-MenuOption -Number '6' -Label 'Run validation suite' -Description 'Run backend and frontend quality gates'
    Write-Host
    Write-Host '  MAINTENANCE & DATA' -ForegroundColor DarkCyan
    Write-MenuOption -Number '7' -Label 'Remove logs' -Description 'Clear generated application logs'
    Write-MenuOption -Number '8' -Label 'Clear cache' -Description 'Remove runtime and test/tool caches'
    Write-MenuOption -Number '9' -Label 'Remove all data' -Description 'Delete user-generated data; preserve application files' -Destructive
    Write-MenuOption -Number '10' -Label 'Uninstall application' -Description 'Remove dependencies and build output' -Destructive
    Write-Host
    Write-Host '  UPDATES' -ForegroundColor DarkCyan
    Write-MenuOption -Number '11' -Label 'Check for updates' -Description 'Report main-branch status without changing files'
    Write-MenuOption -Number '12' -Label 'Update application' -Description 'Pull the latest changes from the main branch'
    Write-Host
    Write-MenuRule
    Write-MenuOption -Number '13' -Label 'Exit' -Description 'Close this launcher' -Exit
    Write-MenuRule
    Write-Host
}

while ($true) {
    Show-LauncherMenu
    $rawSelection = Read-Host '  Select an option (1-13)'
    if ($null -eq $rawSelection) { break }
    $selection = ([string]$rawSelection).Trim()
    if ($selection -notmatch '^(13|1[0-2]|[1-9])$') {
        Write-Warn 'Invalid option. Enter a number from 1 to 13.'
        Wait-ForMenuReturn
        continue
    }
    if ($selection -eq '13') { break }
    try {
        switch ($selection) {
            # Application
            '1' { Invoke-Launch; exit 0 }

            # Setup and validation
            '2' { Invoke-Install -PromptForInstallationType }
            '3' { Invoke-FrontendRebuild }
            '4' { Invoke-BackendManagement }
            '5' { Invoke-DatabaseInitialization }
            '6' { Invoke-TestSuite }

            # Maintenance and data
            '7' { Invoke-RemoveApplicationLogs }
            '8' { Invoke-ClearApplicationCache }
            '9' { Invoke-RemoveAllData }
            '10' { Invoke-UninstallApplication }

            # Updates
            '11' { Invoke-CheckForUpdates }
            '12' { Invoke-UpdateApplication }
        }
    } catch {
        Write-Fatal $_.Exception.Message
        if ([Console]::IsInputRedirected) { exit 1 }
    }
    Wait-ForMenuReturn
}
