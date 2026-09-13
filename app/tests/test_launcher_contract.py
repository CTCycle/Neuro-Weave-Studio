from __future__ import annotations

from pathlib import Path

###############################################################################
def test_launcher_opens_browser_only_after_backend_readiness() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    launcher = (repository_root / "start_on_windows.ps1").read_text(encoding="utf-8")
    launch_start = launcher.index("function Invoke-Launch {")
    launch_end = launcher.index("# Installation, database, and validation actions", launch_start)
    launch_body = launcher[launch_start:launch_end]

    frontend_ready = launch_body.index("Write-Ok 'Frontend is ready.'")
    backend_wait = launch_body.index("$backendHealth", frontend_ready)
    backend_healthy = launch_body.index("Write-Ok 'Backend is healthy.'")
    browser_open = launch_body.index("Open-ApplicationBrowser -Uri $uiUrl")

    assert frontend_ready < backend_wait < backend_healthy < browser_open
