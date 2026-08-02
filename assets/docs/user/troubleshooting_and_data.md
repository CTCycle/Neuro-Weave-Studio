# Troubleshooting and Data

Last updated: 2026-08-02

## The application does not open

Check the launcher-owned logs under `app/resources/runtime/logs/` and confirm
that ports 8000 and 5173 are not occupied by another application. Use
`-Mode Stop` only for processes recorded by NeuroWeave.

## Database errors

If startup reports missing tables, confirm the active `settings/.env` values.
For a disposable or intentionally reset workspace, run:

```powershell
.\start_on_windows.ps1 -Mode InitializeDatabase
```

This is destructive: embedded SQLite data is removed, and PostgreSQL
NeuroWeave-owned tables are recreated. Existing PostgreSQL databases, users,
grants, and unrelated schemas are preserved.

## Save or export errors

- HTTP 409 on save means the project revision is stale; reopen it.
- HTTP 422 means graph validation or framework compatibility failed.
- HTTP 501 from `.keras` export means optional dependencies or binary support
  are missing.
- A successful HTTP response is not evidence that an external provider or
  generated artifact exists; inspect the returned payload and export path.

Never edit the database while the application is running. Use the repository
API, the documented initializer, and exact configured paths.
