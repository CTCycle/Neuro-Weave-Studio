param(
    [ValidateSet("Launch", "Install", "InitializeDatabase", "Dev", "Backend", "Frontend", "Build", "Test", "Stop")]
    [string]$Mode = "Launch",
    [switch]$NoBrowser,
    [switch]$VisibleBackendLog
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$AppRoot = Join-Path $Root "app"
$ServerRoot = Join-Path $AppRoot "server"
$ClientRoot = Join-Path $AppRoot "client"
$SettingsRoot = Join-Path $Root "settings"
$RuntimeRoot = Join-Path $AppRoot "resources\runtime"
$LogRoot = Join-Path $RuntimeRoot "logs"
$PythonRoot = Join-Path $Root "runtimes\python"
$UvRoot = Join-Path $Root "runtimes\uv"
$NodeRoot = Join-Path $Root "runtimes\node"
$VenvRoot = Join-Path $ServerRoot ".venv"
$PidFile = Join-Path $RuntimeRoot "launcher-pids.json"
$PythonVersion = "3.12.11"
$UvVersion = "0.12.0"
$NodeVersion = "22.23.1"
$PythonExe = $null
$UvExe = $null
$NodeExe = $null
$NpmCmd = $null

New-Item -ItemType Directory -Force $RuntimeRoot, $LogRoot | Out-Null

function Ensure-EnvironmentFile {
    $environmentFile = Join-Path $SettingsRoot ".env"
    $exampleFile = Join-Path $SettingsRoot ".env.example"
    if (-not (Test-Path -LiteralPath $environmentFile)) {
        if (-not (Test-Path -LiteralPath $exampleFile)) {
            throw "Missing settings/.env.example."
        }
        Copy-Item -LiteralPath $exampleFile -Destination $environmentFile
    }
}

function Invoke-External {
    param([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory)
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

function Download-Zip {
    param([string]$Uri, [string]$Destination, [string]$ExtractRoot)
    $downloadRoot = Join-Path $RuntimeRoot "downloads"
    New-Item -ItemType Directory -Force $downloadRoot, $ExtractRoot | Out-Null
    $archive = Join-Path $downloadRoot ([IO.Path]::GetFileName($Uri))
    Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $ExtractRoot -Force
}

function Ensure-Uv {
    $UvExe = Join-Path $UvRoot "uv.exe"
    if (Test-Path $UvExe) { return $UvExe }
    New-Item -ItemType Directory -Force $UvRoot | Out-Null
    $installedUv = Get-Command uv -ErrorAction SilentlyContinue
    if ($installedUv -and (& $installedUv.Source --version) -match $UvVersion) {
        Copy-Item -LiteralPath $installedUv.Source -Destination $UvExe -Force
        return $UvExe
    }
    $extractRoot = Join-Path $RuntimeRoot "downloads\uv"
    Download-Zip "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" "" $extractRoot
    $downloaded = Get-ChildItem -LiteralPath $extractRoot -Recurse -Filter "uv.exe" | Select-Object -First 1
    if (-not $downloaded) { throw "Pinned uv $UvVersion was not found in the downloaded archive." }
    Copy-Item -LiteralPath $downloaded.FullName -Destination $UvExe -Force
    return $UvExe
}

function Ensure-Node {
    $NodeExe = Join-Path $NodeRoot "node.exe"
    $NpmCmd = Join-Path $NodeRoot "npm.cmd"
    if ((Test-Path $NodeExe) -and (Test-Path $NpmCmd)) { return @($NodeExe, $NpmCmd) }
    New-Item -ItemType Directory -Force $NodeRoot | Out-Null
    $extractRoot = Join-Path $RuntimeRoot "downloads\node"
    Download-Zip "https://nodejs.org/dist/v$NodeVersion/node-v$NodeVersion-win-x64.zip" "" $extractRoot
    $downloaded = Get-ChildItem -LiteralPath $extractRoot -Recurse -Filter "node.exe" | Select-Object -First 1
    if (-not $downloaded) { throw "Pinned Node.js $NodeVersion was not found in the downloaded archive." }
    Copy-Item -Path (Join-Path $downloaded.DirectoryName "*") -Destination $NodeRoot -Recurse -Force
    if (-not (Test-Path $NpmCmd)) { throw "Pinned Node.js installation did not contain npm.cmd." }
    return @($NodeExe, $NpmCmd)
}

function Ensure-Python {
    param([string]$UvPath)
    $PythonExe = Get-ChildItem -LiteralPath $PythonRoot -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $PythonExe) {
        New-Item -ItemType Directory -Force $PythonRoot | Out-Null
        Invoke-External $UvPath @("python", "install", "--install-dir", $PythonRoot, "--no-bin", "--no-registry", $PythonVersion) $Root
        $PythonExe = Get-ChildItem -LiteralPath $PythonRoot -Recurse -Filter "python.exe" | Select-Object -First 1
    }
    if (-not $PythonExe) { throw "Pinned Python $PythonVersion is not installed under $PythonRoot." }
    return $PythonExe.FullName
}

function Initialize-Tools {
    Ensure-EnvironmentFile
    $script:UvExe = Ensure-Uv
    $node = Ensure-Node
    $script:NodeExe = $node[0]
    $script:NpmCmd = $node[1]
    $script:PythonExe = Ensure-Python $UvExe
}

function Read-ProcessRecords {
    if (-not (Test-Path $PidFile)) { return @() }
    $raw = Get-Content -Raw $PidFile
    if (-not $raw.Trim()) { return @() }
    $records = ConvertFrom-Json $raw
    if ($records -is [array]) { return $records }
    return @($records)
}

function Write-ProcessRecords {
    param([object[]]$Records)
    $Records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PidFile -Encoding utf8
}

function Get-ProcessCommandLine {
    param([int]$TargetPid)
    return Get-CimInstance Win32_Process -Filter "ProcessId = $TargetPid" -ErrorAction SilentlyContinue
}

function Test-OwnedProcess {
    param([object]$Record)
    $process = Get-Process -Id ([int]$Record.pid) -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    $details = Get-ProcessCommandLine ([int]$Record.pid)
    if ($details) {
        return $details.ExecutablePath -eq $Record.executable -and $details.CommandLine -like "*$Root*"
    }
    return $process.Path -eq $Record.executable
}

function Stop-ProjectProcesses {
    $records = @(Read-ProcessRecords)
    foreach ($record in $records) {
        $live = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
        if ($live -and -not (Test-OwnedProcess $record)) {
            throw "Refused to terminate a recorded PID that no longer matches a NeuroWeave process. Inspect $PidFile."
        }
    }
    foreach ($record in $records) {
        if (Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue) {
            Stop-Process -Id ([int]$record.pid) -Force -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path $PidFile) {
        Remove-Item -LiteralPath $PidFile -Force
    }
}

function Test-PortAvailable {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        $owned = (Read-ProcessRecords | Where-Object { [int]$_.pid -eq [int]$connection.OwningProcess })
        if (-not $owned -or -not (Test-OwnedProcess $owned)) {
            throw "Port $Port is already occupied by PID $($connection.OwningProcess). Stop that application or choose another port; NeuroWeave will not kill unrelated processes."
        }
    }
}

function Start-ProjectProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$StandardOutput,
        [string]$StandardError,
        [switch]$RedirectOutput
    )
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $true
    $startInfo.CreateNoWindow = -not $VisibleBackendLog
    if (-not $VisibleBackendLog) { $startInfo.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden }
    $startInfo.Arguments = (($ArgumentList | ForEach-Object { '"' + $_.Replace('"', '\\"') + '"' }) -join ' ')
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) { throw "Could not start $FilePath." }
    return $process
}

function Install-Dependencies {
    Initialize-Tools
    if (-not (Test-Path (Join-Path $ServerRoot "uv.lock"))) {
        Invoke-External $UvExe @("lock") $ServerRoot
    }
    Invoke-External $UvExe @("sync", "--frozen", "--extra", "test", "--python", $PythonExe) $ServerRoot
    if (-not (Test-Path (Join-Path $ClientRoot "package-lock.json"))) {
        Invoke-External $NpmCmd @("install") $ClientRoot
    }
    Invoke-External $NpmCmd @("ci") $ClientRoot
}

function Assert-Ready {
    Initialize-Tools
    if (-not (Test-Path (Join-Path $ServerRoot "uv.lock"))) { throw "Missing app/server/uv.lock. Run Install first." }
    if (-not (Test-Path (Join-Path $ClientRoot "package-lock.json"))) { throw "Missing app/client/package-lock.json. Run Install first." }
    if (-not (Test-Path (Join-Path $VenvRoot "Scripts\python.exe"))) { throw "Missing app/server/.venv. Run Install first." }
    $venvPython = Join-Path $VenvRoot "Scripts\python.exe"
    $basePrefix = (& $venvPython -c "import sys; print(sys.base_prefix)").Trim()
    if (-not $basePrefix.StartsWith($PythonRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "app/server/.venv is not based on the pinned project-local Python under $PythonRoot. Run Install again." }
    Invoke-External $venvPython @("-c", "import fastapi, pydantic, psycopg, sqlalchemy, uvicorn") $ServerRoot
    if (-not (Test-Path (Join-Path $ClientRoot "node_modules\.bin\vite.cmd"))) { throw "Missing Vite runner. Run Install first." }
    if (-not (Test-Path (Join-Path $ClientRoot "dist\index.html"))) { throw "Missing frontend dist. Run Build first." }
}

function Build-Frontend {
    Assert-Ready
    Invoke-External $NpmCmd @("run", "build") $ClientRoot
}

function Start-Backend {
    param([switch]$Reload)
    Test-PortAvailable 8000
    $venvPython = Join-Path $VenvRoot "Scripts\python.exe"
    $env:PYTHONPATH = "$ServerRoot;$VenvRoot\Lib\site-packages"
    $arguments = @("-m", "uvicorn", "neuro_weave.app:app", "--host", "127.0.0.1", "--port", "8000")
    if ($Reload) { $arguments += "--reload" }
    $record = @{ role = "backend"; pid = 0; executable = $PythonExe; startedAt = (Get-Date).ToUniversalTime().ToString("o") }
    if ($VisibleBackendLog) {
        $process = Start-ProjectProcess -FilePath $PythonExe -ArgumentList $arguments -WorkingDirectory $ServerRoot
    } else {
        $stdout = Join-Path $LogRoot "backend.out.log"
        $stderr = Join-Path $LogRoot "backend.err.log"
        $process = Start-ProjectProcess -FilePath $PythonExe -ArgumentList $arguments -WorkingDirectory $ServerRoot -StandardOutput $stdout -StandardError $stderr -RedirectOutput
    }
    $record.pid = $process.Id
    $records = @(Read-ProcessRecords)
    $records += $record
    Write-ProcessRecords $records
}

function Initialize-Database {
    Initialize-Tools
    Stop-ProjectProcesses
    $databaseScript = Join-Path $Root "app\scripts\initialize_database.py"
    if (-not (Test-Path -LiteralPath $databaseScript)) {
        throw "Database initializer is missing: $databaseScript"
    }
    Invoke-External $UvExe @(
        "run", "--project", "app/server", "--python", $PythonExe,
        "python", "app/scripts/initialize_database.py", "--drop-existing"
    ) $Root
    Write-Host "Database initialization completed." -ForegroundColor Green
}

function Start-Frontend {
    param([switch]$Development)
    Test-PortAvailable 5173
    $viteScript = Join-Path $ClientRoot "node_modules\vite\bin\vite.js"
    $arguments = @($viteScript, $(if ($Development) { "dev" } else { "preview" }), "--host", "127.0.0.1", "--port", "5173")
    $record = @{ role = "frontend"; pid = 0; executable = $NodeExe; startedAt = (Get-Date).ToUniversalTime().ToString("o") }
    $stdout = Join-Path $LogRoot "frontend.out.log"
    $stderr = Join-Path $LogRoot "frontend.err.log"
    $process = Start-ProjectProcess -FilePath $NodeExe -ArgumentList $arguments -WorkingDirectory $ClientRoot -StandardOutput $stdout -StandardError $stderr -RedirectOutput
    $record.pid = $process.Id
    $records = @(Read-ProcessRecords)
    $records += $record
    Write-ProcessRecords $records
}

function Wait-ForUrl {
    param([string]$Url)
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "Timed out waiting for $Url. See logs under $LogRoot."
}

function Open-Application {
    Wait-ForUrl "http://127.0.0.1:8000/health"
    Wait-ForUrl "http://127.0.0.1:5173/"
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:5173/" | Out-Null }
}

switch ($Mode) {
    "Install" { Install-Dependencies }
    "InitializeDatabase" { Initialize-Database }
    "Build" { Build-Frontend }
    "Stop" { Stop-ProjectProcesses }
    "Backend" { Assert-Ready; Start-Backend; Wait-ForUrl "http://127.0.0.1:8000/health" }
    "Frontend" { Assert-Ready; Start-Frontend -Development; Wait-ForUrl "http://127.0.0.1:5173/" }
    "Dev" {
        Assert-Ready
        Stop-ProjectProcesses
        Start-Backend -Reload
        Start-Frontend -Development
        Open-Application
    }
    "Launch" {
        Assert-Ready
        Stop-ProjectProcesses
        Build-Frontend
        Start-Backend
        Start-Frontend
        Open-Application
    }
    "Test" {
        Install-Dependencies
        Invoke-External $UvExe @("run", "--frozen", "--extra", "test", "ruff", "check", ".") $ServerRoot
        Invoke-External $UvExe @("run", "--frozen", "--extra", "test", "pytest") $ServerRoot
        Invoke-External $NpmCmd @("run", "lint") $ClientRoot
        Invoke-External $NpmCmd @("run", "test") $ClientRoot
        Invoke-External $NpmCmd @("run", "build") $ClientRoot
    }
}
