[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$script:RepoRoot = $PSScriptRoot
$script:AppDir = Join-Path $RepoRoot 'app'
$script:ServerDir = Join-Path $AppDir 'server'
$script:ClientDir = Join-Path $AppDir 'client'
$script:ScriptsDir = Join-Path $AppDir 'scripts'
$script:SettingsDir = Join-Path $RepoRoot 'settings'
$script:RuntimeDir = Join-Path $RepoRoot 'runtimes'
$script:PythonDir = Join-Path $RuntimeDir 'python'
$script:UvDir = Join-Path $RuntimeDir 'uv'
$script:NodeDir = Join-Path $RuntimeDir 'node'
$script:VenvDir = Join-Path $ServerDir '.venv'
$script:EnvironmentFile = Join-Path $SettingsDir '.env'
$script:EnvironmentExample = Join-Path $SettingsDir '.env.example'
$script:LogDir = Join-Path $AppDir 'resources\runtime\logs'
$script:PythonVersion = '3.12.11'
$script:UvVersion = '0.12.0'
$script:NodeVersion = '22.23.1'
$script:Children = [System.Collections.Generic.List[object]]::new()

function Write-Step([string]$Message) { Write-Host "[STEP] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK]   $Message" -ForegroundColor Green }
function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor DarkCyan }
function Write-Warn([string]$Message) { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Fatal([string]$Message) { Write-Host "[FAIL] $Message" -ForegroundColor Red }

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
    $env:PATH = "$($script:NodeDir);$($env:PATH)"
    Clear-PythonEnvironment
}

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    New-Item -ItemType Directory -Path (Split-Path -Parent $ArchivePath), $DestinationPath -Force | Out-Null
    Write-Info "Downloading runtime archive."
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

function Import-DotEnv {
    if (-not (Test-Path -LiteralPath $script:EnvironmentFile)) {
        if (-not (Test-Path -LiteralPath $script:EnvironmentExample)) {
            throw "Missing environment template: $($script:EnvironmentExample)"
        }
        New-Item -ItemType Directory -Path $script:SettingsDir -Force | Out-Null
        Copy-Item -LiteralPath $script:EnvironmentExample -Destination $script:EnvironmentFile
        Write-Info 'Created settings/.env from settings/.env.example.'
    }

    $defaults = [ordered]@{
        FASTAPI_HOST = '127.0.0.1'
        FASTAPI_PORT = '8000'
        UI_HOST = '127.0.0.1'
        UI_PORT = '5173'
        VITE_API_BASE_URL = '/api'
        RELOAD = 'true'
    }
    foreach ($key in $defaults.Keys) {
        if (-not (Test-Path -LiteralPath "Env:$key")) {
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
        if (-not (Test-Path -LiteralPath "Env:$key")) {
            Set-Item -LiteralPath "Env:$key" -Value $value
        }
    }
}

function Sync-Dependencies {
    param([switch]$IncludeTests)

    $uv = Get-Uv
    $python = Get-PortablePython
    Set-LauncherEnvironment
    $uvArguments = @('sync', '--python', $python)
    if ($IncludeTests) { $uvArguments += @('--extra', 'test') }
    Push-Location $script:ServerDir
    try {
        Write-Step 'Synchronizing backend dependencies.'
        & $uv @uvArguments
        if ($LASTEXITCODE -ne 0) { throw "Backend dependency synchronization failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }

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

function Test-DependenciesReady {
    param([switch]$IncludeTests)

    $venvPython = Get-VenvPython
    $clientPackage = Join-Path $script:ClientDir 'package.json'
    $clientLock = Join-Path $script:ClientDir 'package-lock.json'
    $clientModules = Join-Path $script:ClientDir 'node_modules'
    $clientInstallState = Join-Path $clientModules '.package-lock.json'
    $vite = Join-Path $clientModules '.bin\vite.cmd'
    if (-not (Test-Path -LiteralPath $venvPython) -or
        -not (Test-Path -LiteralPath (Join-Path $script:ServerDir 'app.py')) -or
        -not (Test-Path -LiteralPath $clientPackage) -or
        -not (Test-Path -LiteralPath $clientLock) -or
        -not (Test-Path -LiteralPath $clientInstallState) -or
        -not (Test-Path -LiteralPath $vite)) {
        return $false
    }
    $imports = 'import fastapi, uvicorn, sqlalchemy, psycopg'
    if ($IncludeTests) { $imports += ', pytest, ruff' }
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
        [int]$Attempts = 60,
        [int]$IntervalSeconds = 1
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) { return $true }
        } catch { }
        if ($attempt -lt $Attempts) { Start-Sleep -Seconds $IntervalSeconds }
    }
    return $false
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

function Stop-ManagedProcesses {
    for ($index = $script:Children.Count - 1; $index -ge 0; $index--) {
        $child = $script:Children[$index]
        if (-not $child.Process.HasExited) {
            & taskkill.exe /PID $child.Process.Id /T /F *> $null
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
    Invoke-FrontendBuild

    $backendHost = [Environment]::GetEnvironmentVariable('FASTAPI_HOST', 'Process')
    $backendPort = Get-ConfiguredInt -Name 'FASTAPI_PORT' -Default '8000'
    $uiHost = [Environment]::GetEnvironmentVariable('UI_HOST', 'Process')
    $uiPort = Get-ConfiguredInt -Name 'UI_PORT' -Default '5173'
    Assert-PortAvailable -Port $backendPort
    Assert-PortAvailable -Port $uiPort

    $venvPython = Get-VenvPython
    $npm = Get-Npm
    $backendArguments = @('-m', 'uvicorn', 'app:app', '--host', $backendHost, '--port', [string]$backendPort)
    if ([Environment]::GetEnvironmentVariable('RELOAD', 'Process') -ieq 'true') { $backendArguments += '--reload' }
    try {
        Write-Step "Starting backend at http://$backendHost`:$backendPort"
        Start-ManagedProcess -Name 'Backend' -FilePath $venvPython -ArgumentList $backendArguments -WorkingDirectory $script:ServerDir | Out-Null
        $backendHealth = "http://$backendHost`:$backendPort/health"
        if (-not (Invoke-HealthCheck -Url $backendHealth)) {
            throw "Backend did not become healthy at $backendHealth. See $($script:LogDir)."
        }
        Write-Ok 'Backend is healthy.'

        Write-Step "Starting frontend at http://$uiHost`:$uiPort"
        Start-ManagedProcess -Name 'Frontend' -FilePath $npm -ArgumentList @('run', 'preview', '--', '--host', $uiHost, '--port', [string]$uiPort, '--strictPort') -WorkingDirectory $script:ClientDir | Out-Null
        $uiUrl = "http://$uiHost`:$uiPort/"
        if (-not (Invoke-HealthCheck -Url $uiUrl)) {
            throw "Frontend did not become ready at $uiUrl. See $($script:LogDir)."
        }
        Write-Ok 'Frontend is ready.'
        Write-Host "NeuroWeave is running. Backend: $backendHealth | Frontend: $uiUrl" -ForegroundColor Green
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

function Invoke-Install {
    Ensure-PortableRuntimes
    Import-DotEnv
    Sync-Dependencies -IncludeTests
    Invoke-FrontendBuild
}

function Invoke-PostgresInitialization {
    Ensure-PortableRuntimes
    Import-DotEnv
    if (-not (Test-DependenciesReady)) { Sync-Dependencies }
    $venvPython = Get-VenvPython
    $initializer = Join-Path $script:ScriptsDir 'initialize_database.py'
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $script:ServerDir
        Write-Step 'Running explicit PostgreSQL database and schema initialization.'
        Push-Location $script:ServerDir
        try {
            & $venvPython $initializer
            if ($LASTEXITCODE -ne 0) { throw "PostgreSQL initialization failed with exit code $LASTEXITCODE." }
        } finally { Pop-Location }
    } finally {
        if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $previousPythonPath }
    }
    Write-Ok 'PostgreSQL initialization completed.'
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
            Write-Step 'Running backend Ruff and pytest.'
            & $venvPython -m ruff check . ..\tests
            if ($LASTEXITCODE -ne 0) { throw "Backend Ruff failed with exit code $LASTEXITCODE." }
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

function Show-Menu {
    Write-Host
    Write-Host 'NeuroWeave Studio' -ForegroundColor Cyan
    Write-Host 'Local workspace control center' -ForegroundColor DarkGray
    Write-Host
    Write-Host '  [1] Launch application' -ForegroundColor White
    Write-Host '  [2] Install or repair dependencies' -ForegroundColor White
    Write-Host '  [3] Initialize PostgreSQL' -ForegroundColor White
    Write-Host '  [4] Run validation suite' -ForegroundColor White
    Write-Host '  [5] Exit' -ForegroundColor White
}

while ($true) {
    Show-Menu
    $rawSelection = Read-Host 'Select an option [1-5]'
    if ($null -eq $rawSelection) { break }
    $selection = ([string]$rawSelection).Trim()
    if ($selection -eq '5') { break }
    try {
        switch ($selection) {
            '1' { Invoke-Launch }
            '2' { Invoke-Install }
            '3' { Invoke-PostgresInitialization }
            '4' { Invoke-TestSuite }
            default { throw 'Select a number from 1 through 5.' }
        }
    } catch {
        Write-Fatal $_.Exception.Message
    }
    Write-Host
}
