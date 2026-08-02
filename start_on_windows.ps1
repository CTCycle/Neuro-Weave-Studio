param(
    [ValidateSet("Dev", "Backend", "Frontend", "Test", "Build")]
    [string]$Mode = "Dev"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Server = Join-Path $Root "app\server"
$Frontend = Join-Path $Root "frontend"

function Start-Backend {
    Push-Location $Server
    $PreviousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $Server
        uv sync --extra test
        uv run uvicorn app:app --reload --port 8000
    } finally {
        if ($null -eq $PreviousPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $PreviousPythonPath
        }
        Pop-Location
    }
}

function Start-Frontend {
    Push-Location $Frontend
    try {
        if (-not (Test-Path "node_modules")) { npm install }
        npm run dev
    } finally { Pop-Location }
}

switch ($Mode) {
    "Backend" { Start-Backend }
    "Frontend" { Start-Frontend }
    "Dev" {
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $MyInvocation.MyCommand.Path, "-Mode", "Backend"
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $MyInvocation.MyCommand.Path, "-Mode", "Frontend"
    }
    "Test" {
        Push-Location $Server
        $PreviousPythonPath = $env:PYTHONPATH
        try {
            $env:PYTHONPATH = $Server
            uv sync --extra test
            uv run ruff check . ..\tests
            uv run pytest ..\tests
        } finally {
            if ($null -eq $PreviousPythonPath) {
                Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
            } else {
                $env:PYTHONPATH = $PreviousPythonPath
            }
            Pop-Location
        }
        Push-Location $Frontend; npm install; npm run lint; npm run test; Pop-Location
    }
    "Build" {
        Push-Location $Frontend; npm install; npm run build; Pop-Location
    }
}
