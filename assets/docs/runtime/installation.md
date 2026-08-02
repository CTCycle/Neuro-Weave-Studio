# Installation

Last updated: 2026-08-02

Install Python 3.12 or newer, Node.js 20 or newer, and `uv`, or use the pinned
project-local runtimes through `start_on_windows.bat -Mode Install`.

Manual setup:

```powershell
cd app/server
uv sync --frozen --extra test
cd ../client
npm ci
```

The backend dependency set includes SQLAlchemy and `psycopg[binary]` for the
SQLite/PostgreSQL contract. Add `--extra keras` only when local TensorFlow-backed
binary export is required. Copy `settings/.env.example` to `settings/.env`
only when overriding defaults; the Windows launcher performs that copy
non-destructively when the file is absent.
