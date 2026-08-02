# Installation

Last updated: 2026-08-02

Install Python 3.12 or newer, Node.js 20 or newer, and `uv`, or use the pinned project-local runtimes through `start_on_windows.bat -Mode Install`. Run `uv sync --frozen --extra test` in `app/server` and `npm ci` in `app/client`. Add `--extra keras` only when local TensorFlow-backed binary export is required because it is a large optional dependency. Copy `settings/.env.example` to `settings/.env` only when overriding defaults.
