@echo off
setlocal
set "SCRIPT=%~dp0start_on_windows.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
