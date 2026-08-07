# Installation

Last updated: 2026-08-06

Run `start_on_windows.ps1` from the repository root and select **Install or repair dependencies** when local runtimes or dependencies are missing. The launcher asks for a **Development** profile (including Ruff, Pyright, and pytest) or a **Standard** profile (runtime dependencies only). It uses pinned repository-local Python 3.12.11, uv 0.12.0, and Node.js 22.23.1 under `runtimes/`; subsequent launches validate the installed environment and skip synchronization when it is usable.

Keras 3 is a required core dependency. The install action accepts an exact comma-separated execution-backend set (`tensorflow`, `pytorch`, `jax`) including `none`; the deterministic form is `start_on_windows.ps1 -Action Install -Backends tensorflow,pytorch,jax`. The JAX extra installs bounded `jax` and `flax` packages. Normal launch preserves already-installed extras, while an explicit install set is authoritative. Core-only mode supports source generation but exposes no operational binary backend until a probe succeeds.

The canonical environment file is `settings/.env`. The shared environment loader creates it from `settings/.env.example` when it is absent, for both launcher and direct backend startup, without overwriting an existing file. `start_on_windows.ps1 -Action Install -InstallationType Standard -Backends tensorflow` provides a deterministic non-interactive standard installation; omit `-InstallationType Standard` when the development tooling is desired. Never commit `settings/.env` or other `.env` files.
