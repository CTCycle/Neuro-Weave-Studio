**********************
Avvio della trascrizione di PowerShell
Ora di inizio: 20260928225557
Nome utente: BLUEGENE\TV
Utente RunAs: BLUEGENE\TV
Nome configurazione:
Computer: BLUEGENE (Microsoft Windows NT 10.0.26200.0)
Applicazione host: C:\Program Files\PowerShell\7\pwsh.dll -Command try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}
$log = 'assets/QA/comprehensive-validation-20260928/logs/process-cleanup-nw-t3-05-final.md'; try { Start-Transcript -Path $log -Force | Out-Null; Write-Output 'NW-T3-05 final owned-process cleanup'; Write-Output 'Before:'; Get-NetTCPConnection -LocalPort 9750,6425 -State Listen | Select-Object LocalPort,OwningProcess,State; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @(21084,8600) } | ForEach-Object { "PID=$($_.ProcessId) $($_.CommandLine)" }; Stop-Process -Id 21084,8600 -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2; Write-Output 'After:'; Get-NetTCPConnection -LocalPort 9750,6425 -State Listen -ErrorAction SilentlyContinue | Select-Object LocalPort,OwningProcess,State; Write-Output 'Owned PIDs still present:'; Get-Process -Id 21084,8600 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName } finally { Stop-Transcript | Out-Null }
ID processo: 13284
PSVersion: 7.6.6
PSEdition: Core
GitCommitId: 7.6.6
OS: Microsoft Windows 10.0.26200
Platform: Win32NT
PSCompatibleVersions: 1.0, 2.0, 3.0, 4.0, 5.0, 5.1, 6.0, 7.0
PSRemotingProtocolVersion: 2.4
SerializationVersion: 1.1.0.1
WSManStackVersion: 3.0
**********************
NW-T3-05 final owned-process cleanup
Before:

LocalPort OwningProcess  State
--------- -------------  -----
     9750         21084 Listen
     6425          8600 Listen
PID=21084 "G:\Projects\Repositories\Active projects\Neuro Weave Studio\app\server\.venv\Scripts\python.exe"  -m uvicorn app:create_app --factory --host 127.0.0.1 --port 9750
PID=8600 node  tools/preview-server.mjs --host 127.0.0.1 --port 6425 --strictPort
After:
Owned PIDs still present:
**********************
Fine della trascrizione di PowerShell
Ora di fine: 20260928225600
**********************
