# Installation

Last updated: 2026-08-05

Run `start_on_windows.ps1` from the repository root and select **Install or repair dependencies** when local runtimes or dependencies are missing. The launcher uses pinned repository-local Python 3.12.11, uv 0.12.0, and Node.js 22.23.1 under `runtimes/`; subsequent launches validate the installed environment and skip synchronization when it is usable.

Keras is a required core dependency. The install menu chooses the optional runtime extras: Keras only, Keras plus TensorFlow, Keras plus Torch, or both. Core-only mode supports source generation but intentionally has no operational binary-export backend. Dependency synchronization preserves already-installed TensorFlow/Torch extras unless a new install mode is selected. For manual setup use `uv sync --extra test`, then add `--extra tensorflow` or `--extra torch` as needed.

The canonical environment file is `settings/.env`. The shared environment loader creates it from `settings/.env.example` when it is absent, for both launcher and direct backend startup, without overwriting an existing file. Never commit `settings/.env` or other `.env` files.
