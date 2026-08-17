[CmdletBinding()]
param(
    [ValidateSet('Menu', 'Install', 'Launch', 'InitDb', 'Test', 'CleanLogs', 'ClearCache', 'Uninstall')]
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
$script:ServerDir = Join-Path $AppDir 'server'
$script:ClientDir = Join-Path $AppDir 'client'
$script:ScriptsDir = Join-Path $AppDir 'scripts'
$script:EnvironmentBootstrap = Join-Path $ScriptsDir 'ensure_environment.py'
$script:RuntimeDir = Join-Path $RepoRoot 'runtimes'
$script:PythonDir = Join-Path $RuntimeDir 'python'
$script:UvDir = Join-Path $RuntimeDir 'uv'
$script:NodeDir = Join-Path $RuntimeDir 'node'
$script:VenvDir = Join-Path $ServerDir '.venv'
$script:EnvironmentFile = Join-Path (Join-Path $RepoRoot 'settings') '.env'
$script:LogDir = Join-Path $AppDir 'resources\runtime\logs'
$script:BackendInstallState = Join-Path $AppDir 'resources\runtime\backend-installation.json'
$script:FrontendDistIndex = Join-Path $ClientDir 'dist\index.html'
$script:PythonVersion = '3.12.11'
$script:UvVersion = '0.12.0'
$script:NodeVersion = '22.23.1'
$script:Children = [System.Collections.Generic.List[object]]::new()
$script:NoBrowser = $NoBrowser

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
    $env:UV_CACHE_DIR = Join-Path $script:RuntimeDir '.uv-cache'
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
    $pythonVersion = (& $python --version).Trim()
    if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne "Python $($script:PythonVersion)") {
        throw "Expected Python $($script:PythonVersion), found '$pythonVersion'."
    }

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
        ALWAYS_REBUILD = 'true'
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

function Get-InstalledExecutionBackends {
    $venvPython = Get-VenvPython
    if (-not (Test-Path -LiteralPath $venvPython)) { return @() }
    $probe = "import importlib.util; print(','.join(name for name in ('tensorflow', 'torch', 'jax') if importlib.util.find_spec(name)))"
    $installed = (& $venvPython -c $probe 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($installed)) { return @() }
    return @($installed -split ',' | ForEach-Object {
            $name = $_.Trim()
            if ($name -eq 'torch') { 'pytorch' } else { $name }
        } | Where-Object { $_ })
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
        [Parameter(Mandatory = $true)][string[]]$Backends,
        [Parameter(Mandatory = $true)][string]$Acceleration,
        [string[]]$Failures = @()
    )
    New-Item -ItemType Directory -Path (Split-Path -Parent $script:BackendInstallState) -Force | Out-Null
    [ordered]@{
        schema_version = 1
        backends = @($Backends)
        acceleration = $Acceleration
        failures = @($Failures)
        recorded_at = [DateTime]::UtcNow.ToString('O')
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $script:BackendInstallState -Encoding UTF8
}

function ConvertTo-BackendExtras {
    param([AllowEmptyString()][string]$Selection)

    if ([string]::IsNullOrWhiteSpace($Selection)) { return @('auto') }
    $requested = @($Selection -split ',' | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
    if ($requested -contains 'auto') {
        if ($requested.Count -gt 1) { throw "The backend selection 'auto' cannot be combined with another backend." }
        return @('auto')
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
        throw "Unknown execution backend(s): $($invalid -join ', '). Choose tensorflow, pytorch, and/or jax."
    }
    return @($requested | Select-Object -Unique)
}

function Read-BackendSelection {
    $installed = @(Get-InstalledExecutionBackends)
    Write-Host "Installed/available execution backends: $($(if ($installed.Count) { $installed -join ', ' } else { 'none' }))" -ForegroundColor DarkCyan
    $nvidia = Get-NvidiaCudaInfo
    $recommendation = if ($nvidia.Available) { 'pytorch with CUDA when supported' } else { 'pytorch CPU' }
    Write-Host "Recommended first setup: $recommendation. Choose 'all' for TensorFlow + PyTorch + JAX, or 'none' for Keras-only." -ForegroundColor DarkGray
    $selection = (Read-Host 'Execution backends to install (Enter=auto, all, tensorflow,pytorch,jax, or none)').Trim().ToLowerInvariant()
    return @(ConvertTo-BackendExtras -Selection $selection)
}

function Resolve-BackendExtras {
    param([string[]]$Backends = @())

    $resolved = [System.Collections.Generic.List[string]]::new()
    foreach ($backend in @($Backends)) {
        $normalized = ([string]$backend).Trim().ToLowerInvariant()
        if (-not $normalized -or $normalized -eq 'none') { continue }
        if ($normalized -eq 'auto') { $normalized = 'pytorch' }
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
            $previous = [Environment]::GetEnvironmentVariable('KERAS_BACKEND', 'Process')
            try {
                $value = if ($backend -eq 'pytorch') { 'torch' } else { $backend }
                $env:KERAS_BACKEND = $value
                $raw = (& $venvPython -m models.keras_worker --probe 2>$null | Select-Object -Last 1)
                if ($raw) {
                    try { $reports.Add(($raw | ConvertFrom-Json)) } catch { }
                }
            } finally {
                if ($null -eq $previous) { Remove-Item Env:KERAS_BACKEND -ErrorAction SilentlyContinue } else { $env:KERAS_BACKEND = $previous }
            }
        }
    } finally { Pop-Location }
    return @($reports.ToArray())
}

function Write-BackendCapabilitySummary {
    $nvidia = Get-NvidiaCudaInfo
    if ($nvidia.Available) {
        Write-Info "NVIDIA GPU detected: $($nvidia.DeviceName) (CUDA driver $($nvidia.CudaVersion))."
    } else {
        Write-Info 'No operational NVIDIA CUDA device was detected; GPU-capable backends will use CPU fallbacks.'
    }
    foreach ($report in @(Get-BackendCapabilityReport)) {
        $label = if ($report.available) { "$($report.backend) / $($report.accelerator) / $($report.device_name)" } else { "$($report.backend) / unavailable" }
        Write-Status -Level $(if ($report.available) { 'OK' } else { 'WARN' }) -Message "Backend capability: $label"
    }
}

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
    Push-Location $script:ClientDir
    try {
        Write-Step 'Building frontend.'
        & $npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
}

function Invoke-OptionalBackendSetup {
    $state = Get-BackendInstallationState
    if ($null -ne $state) {
        Write-Ok 'Backend setup has already been acknowledged; skipped optional dependency installation.'
        Write-BackendCapabilitySummary
        return
    }

    $selection = if ([string]::IsNullOrWhiteSpace($Backends)) { @(Read-BackendSelection) } else {
        @(ConvertTo-BackendExtras -Selection $Backends)
    }
    if ($selection -contains 'auto') { $selection = @('pytorch') }
    if ($selection.Count -eq 0) {
        Write-Info 'Keras-only mode selected. Source generation remains available; inference backends can be added later with Install.'
        Set-BackendInstallationState -Backends @() -Acceleration $script:Acceleration
        return
    }

    $failures = [System.Collections.Generic.List[string]]::new()
    $successful = [System.Collections.Generic.List[string]]::new()
    Write-Step "Installing selected execution backends independently: $($selection -join ', ')."
    foreach ($backend in @($selection)) {
        $candidate = @($successful.ToArray()) + @($backend)
        try {
            Sync-Dependencies -BackendExtras $candidate -ExplicitBackendSet -SkipFrontend
            $successful.Add($backend)
        } catch {
            $message = $_.Exception.Message
            if ($backend -eq 'pytorch' -and (Get-PytorchBackendExtra) -eq 'torch-cu126') {
                Write-Warn 'PyTorch CUDA installation failed; retrying that backend with the CPU wheel.'
                try {
                    $script:Acceleration = 'CPU'
                    Sync-Dependencies -BackendExtras (@($successful.ToArray()) + @($backend)) -ExplicitBackendSet -SkipFrontend
                    $successful.Add($backend)
                    Write-Ok 'PyTorch CPU fallback installed.'
                    continue
                } catch {
                    $message = $_.Exception.Message
                }
            }
            $failures.Add("${backend}: $message")
            Write-Warn "Optional backend '$backend' was skipped: $message"
        }
    }
    Sync-Dependencies -BackendExtras @($successful.ToArray()) -ExplicitBackendSet
    if ($failures.Count -gt 0) {
        Write-Warn "Continuing with the core application. Failed optional backend(s): $($failures -join '; ')"
    }
    Set-BackendInstallationState -Backends $selection -Acceleration $script:Acceleration -Failures @($failures.ToArray())
    Write-BackendCapabilitySummary
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
    & $uv --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $node --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    $imports = "import fastapi, uvicorn, sqlalchemy; import importlib.metadata; importlib.metadata.version('keras')"
    if ($IncludeTests) { $imports += "; importlib.metadata.version('pytest'); importlib.metadata.version('ruff'); importlib.metadata.version('pyright')" }
    & $venvPython -c $imports *> $null
    return $LASTEXITCODE -eq 0
}

function Get-ConfiguredInt {
    param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][string]$Default)
    $raw = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($raw)) { $raw = $Default }
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

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    New-Item -ItemType Directory -Path $script:LogDir -Force | Out-Null
    $safeName = $Name.ToLowerInvariant()
    $stdout = Join-Path $script:LogDir "$safeName.out.log"
    $stderr = Join-Path $script:LogDir "$safeName.err.log"
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $script:Children.Add([pscustomobject]@{ Name = $Name; Process = $process; Stdout = $stdout; Stderr = $stderr })
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

function Invoke-Launch {
    Ensure-PortableRuntimes
    Import-DotEnv
    if (-not (Test-DependenciesReady)) {
        Sync-Dependencies
    } else {
        Write-Ok 'Application environments are ready; skipped dependency installation.'
    }
    Invoke-OptionalBackendSetup
    $alwaysRebuild = Get-BooleanEnvironmentValue -Name 'ALWAYS_REBUILD' -Default $true
    if ($alwaysRebuild -or -not (Test-Path -LiteralPath $script:FrontendDistIndex)) {
        if (-not $alwaysRebuild) { Write-Info 'Frontend build output is missing; rebuilding despite ALWAYS_REBUILD=false.' }
        Invoke-FrontendBuild
    } else {
        Write-Ok 'Frontend build output is ready; skipped rebuild because ALWAYS_REBUILD=false.'
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
        $backendProcess = Start-ManagedProcess -Name 'Backend' -FilePath $venvPython -ArgumentList $backendArguments -WorkingDirectory $script:ServerDir
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
        Open-ApplicationBrowser -Uri $uiUrl
        Write-Status -Level SUCCESS -Message 'NeuroWeave Studio is running.'
        Write-Host "  Backend:  $backendHealth (PID $($backendProcess.Id))"
        Write-Host "  Frontend: $uiUrl (PID $($frontendProcess.Id))"
        Write-Host "  Logs:     $($script:LogDir)" -ForegroundColor DarkGray
        Write-Host 'Press Ctrl+C to stop both services cleanly.' -ForegroundColor DarkGray

        while ($true) {
            foreach ($child in @($script:Children)) {
                if ($child.Process.HasExited) {
                    throw "$($child.Name) exited with code $($child.Process.ExitCode). See $($child.Stderr)."
                }
            }
            Start-Sleep -Milliseconds 500
        }
    } finally {
        Stop-ManagedProcesses
    }
}

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
    $selectedInstallationType = $InstallationType
    if ($PromptForInstallationType) { $selectedInstallationType = Read-InstallationType }
    $backendExtras = if ([string]::IsNullOrWhiteSpace($Backends)) { Read-BackendSelection } else {
        @(ConvertTo-BackendExtras -Selection $Backends)
    }
    $includeTests = $selectedInstallationType -eq 'Development'
    Write-Status -Level INFO -Message "Installation profile: $selectedInstallationType"
    Sync-Dependencies -IncludeTests:$includeTests -BackendExtras $backendExtras -ExplicitBackendSet
    $effectiveBackends = if ($backendExtras -contains 'auto') { @('pytorch') } else { @($backendExtras) }
    Set-BackendInstallationState -Backends $effectiveBackends -Acceleration $script:Acceleration
    Write-BackendCapabilitySummary
    Invoke-FrontendBuild
}

function Invoke-DatabaseInitialization {
    Ensure-PortableRuntimes
    Import-DotEnv
    if (-not (Test-DependenciesReady)) { Sync-Dependencies }
    $venvPython = Get-VenvPython
    $initializer = Join-Path $script:ScriptsDir 'initialize_database.py'
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $script:ServerDir
        $reset = Read-Host 'This recreates the managed database and deletes its data. Type RESET to continue'
        $arguments = @()
        if ($reset -ceq 'RESET') {
            $arguments = @('--reset')
            Write-Step 'Recreating the managed database schema from scratch.'
        } else {
            Write-Step 'Checking or initializing the canonical database schema without reset.'
        }
        Push-Location $script:ServerDir
        try {
            & $venvPython $initializer @arguments
            if ($LASTEXITCODE -ne 0) { throw "Database initialization failed with exit code $LASTEXITCODE." }
        } finally { Pop-Location }
    } finally {
        if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $previousPythonPath }
    }
    Write-Ok 'Database initialization completed.'
}

function Invoke-TestSuite {
    Ensure-PortableRuntimes
    Import-DotEnv
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
            & $venvPython -m pytest ..\tests --basetemp (Join-Path $script:RepoRoot 'assets\QA\pytest-tmp-launcher')
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
    $logs | Remove-Item -Force
    Write-Status -Level SUCCESS -Message "Removed $($logs.Count) application log file(s)."
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
    foreach ($cachePath in $cachePaths) {
        Remove-Item -LiteralPath $cachePath.FullName -Recurse -Force
    }
    return [int]$cachePaths.Count
}

function Invoke-ClearApplicationCache {
    Write-Status -Level STEP -Message 'Removing Python bytecode and the repository-local uv cache.'
    $pythonCacheCount = Remove-PythonCaches
    $uvCache = Join-Path $script:RuntimeDir '.uv-cache'
    $uvCacheRemoved = $false
    if (Test-Path -LiteralPath $uvCache) {
        Remove-Item -LiteralPath $uvCache -Recurse -Force
        $uvCacheRemoved = $true
    }
    Write-Status -Level SUCCESS -Message "Caches cleared ($pythonCacheCount Python cache director$(if ($pythonCacheCount -eq 1) { 'y' } else { 'ies' })$(if ($uvCacheRemoved) { ' and uv cache' } else { '' }))."
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
        (Join-Path $script:RuntimeDir '.uv-cache'),
        $script:VenvDir,
        (Join-Path $script:ClientDir 'node_modules'),
        (Join-Path $script:ClientDir 'dist'),
        $script:BackendInstallState
    )
    Write-Status -Level STEP -Message 'Removing generated runtimes, dependencies, and frontend build output.'
    foreach ($target in $targets) {
        if (Test-Path -LiteralPath $target) {
            Write-Info "Removing $target"
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
    $pythonCacheCount = Remove-PythonCaches
    New-Item -ItemType Directory -Path $script:RuntimeDir -Force | Out-Null
    Write-Status -Level SUCCESS -Message "Application dependencies removed. Settings, lockfiles, database files, exports, and $pythonCacheCount Python cache director$(if ($pythonCacheCount -eq 1) { 'y' } else { 'ies' }) were preserved or cleared."
}

if ($Action -ne 'Menu') {
    try {
        switch ($Action) {
            'Install' { Invoke-Install }
            'Launch' { Invoke-Launch }
            'InitDb' { Invoke-DatabaseInitialization }
            'Test' { Invoke-TestSuite }
            'CleanLogs' { Invoke-RemoveApplicationLogs }
            'ClearCache' { Invoke-ClearApplicationCache }
            'Uninstall' { Invoke-UninstallApplication }
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
    Write-MenuOption -Number '2' -Label 'Install or repair dependencies' -Description 'Choose profile, runtimes, and backend extras'
    Write-MenuOption -Number '3' -Label 'Initialize or recreate database' -Description 'Create schema or confirm a destructive reset'
    Write-MenuOption -Number '4' -Label 'Run validation suite' -Description 'Run backend and frontend quality gates'
    Write-Host
    Write-Host '  MAINTENANCE' -ForegroundColor DarkCyan
    Write-MenuOption -Number '5' -Label 'Remove logs' -Description 'Clear generated application logs'
    Write-MenuOption -Number '6' -Label 'Clear cache' -Description 'Remove Python bytecode and uv cache'
    Write-MenuOption -Number '7' -Label 'Uninstall application' -Description 'Remove dependencies and build output' -Destructive
    Write-Host
    Write-MenuRule
    Write-MenuOption -Number '8' -Label 'Exit' -Description 'Close this launcher' -Exit
    Write-MenuRule
    Write-Host
}

while ($true) {
    Show-LauncherMenu
    $rawSelection = Read-Host '  Select an option (1-8)'
    if ($null -eq $rawSelection) { break }
    $selection = ([string]$rawSelection).Trim()
    if ($selection -notmatch '^[1-8]$') {
        Write-Warn 'Invalid option. Enter a number from 1 to 8.'
        Wait-ForMenuReturn
        continue
    }
    if ($selection -eq '8') { break }
    try {
        switch ($selection) {
            '1' { Invoke-Launch }
            '2' { Invoke-Install -PromptForInstallationType }
            '3' { Invoke-DatabaseInitialization }
            '4' { Invoke-TestSuite }
            '5' { Invoke-RemoveApplicationLogs }
            '6' { Invoke-ClearApplicationCache }
            '7' { Invoke-UninstallApplication }
        }
    } catch {
        Write-Fatal $_.Exception.Message
        if ([Console]::IsInputRedirected) { exit 1 }
    }
    Wait-ForMenuReturn
}
