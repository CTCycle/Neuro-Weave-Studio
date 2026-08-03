# Installation

Last updated: 2026-08-03

Run `start_on_windows.ps1` from the repository root and select **Install or repair dependencies** when local runtimes or dependencies are missing. The launcher uses pinned repository-local Python 3.12.11, uv 0.12.0, and Node.js 22.23.1 under `runtimes/`; subsequent launches validate the installed environment and skip synchronization when it is usable.

The canonical environment file is `settings/.env`. If it is absent, the launcher copies `settings/.env.example` without overwriting an existing file. Never commit `settings/.env` or other `.env` files.
