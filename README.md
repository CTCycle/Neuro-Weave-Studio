<p align="center">
  <img src="assets/brand/neuroweave-mark.svg" alt="NeuroWeave Studio" width="96" />
</p>

<h1 align="center">NeuroWeave Studio</h1>

<p align="center">
  <strong>Design, validate, save, and export machine-learning model graphs in a local visual workspace.</strong>
</p>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white&style=flat-square" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/Angular-22-DD0031?logo=angular&logoColor=white&style=flat-square" alt="Angular 22" />
  <img src="https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D4?logo=windows&logoColor=white&style=flat-square" alt="Windows" />
</p>

## Overview

NeuroWeave Studio is a locally hosted visual editor for designing, validating, saving, and exporting machine-learning model graphs. Its Angular 22 interface and FastAPI service run on your own computer, with project metadata and graph revisions stored in SQLite. You build models as validated directed acyclic graphs from a node catalog, configure parameters, connect ports, and review graph and tensor-shape diagnostics. The application supports synthetic, forward-only functional validation, reusable custom node templates, external dataset registration, and source or bundle export when a runtime is available. Training, prediction serving, dataset downloads, and preprocessing are intentionally out of scope, so exported weights remain initialized and untrained. The Windows launcher provisions pinned Python, uv, and Node.js runtimes so no separate toolchain installation is required.

## Key Features

- Create model graphs from the node catalog, configure parameters, connect ports, and review graph and tensor-shape diagnostics.
- Save projects and graph revisions, and create reusable custom node templates.
- Register external datasets and their metadata without copying or downloading dataset contents.
- Run synthetic, forward-only functional validation and tensor inspection against an installed execution profile.
- Generate model source, project bundles, and supported model-format exports when the required runtime is available.
- Manage optional execution backends (PyTorch, TensorFlow, JAX) through the launcher, with CPU fallback and probe-based capability reporting.
- Inspect environment, backend, device, and policy controls from the Environment page.

## Screenshots

No product screenshots are bundled yet. This section will be populated as stable captures become available. The visual-parity captures under `assets/QA/` are regression artifacts, not documentation screenshots.

## Tech Stack

| Component | Technology |
| --- | --- |
| Frontend | Angular 22 (TypeScript) |
| Backend | FastAPI on Python 3.13 |
| Persistence | SQLite (SQLAlchemy, Alembic) |
| ML core | Keras 3 |
| Optional backends | PyTorch, TensorFlow, JAX |
| Runtimes | Pinned Python 3.13.14, uv 0.12.0, Node.js 22.23.1 |
| Platform | Windows 10/11 (64-bit) |

## Installation

### Launcher (recommended)

The supported quick-start targets 64-bit Windows 10 or 11 with PowerShell 7. Git and internet access are needed for cloning and the initial runtime and dependency installation; separate Python and Node.js installations are not required.

```powershell
git clone https://github.com/CTCycle/Neuro-Weave-Studio.git
Set-Location .\Neuro-Weave-Studio
.\start_on_windows.ps1 -Action Install -InstallationType Standard
.\start_on_windows.ps1 -Action Launch
```

The first install creates `settings/.env` from `settings/.env.example`, initializes or upgrades the SQLite database, and builds the frontend. The launcher starts the local services and opens the app in your default browser when possible. The backend serves the built frontend on the same origin (default port `9850`) and stays attached to the launcher console; press Enter or Ctrl+C there to stop the application. Use `-InstallationType Development` to also install Ruff, Pyright, and pytest. Optional execution backends can be selected later through **Manage compute backends** in the launcher.

### Manual

For separate process control, follow the manual development startup in [assets/docs/runtime/startup.md](assets/docs/runtime/startup.md). In short: run `uv run uvicorn app:create_app --factory` from `app/server` and `npm run dev` from `app/client`, with the cache environment variables documented there. Keep `settings/.env` in place for consistent host and port configuration.

## Configuration

Keep machine-specific values in the ignored `settings/.env` file; the launcher creates it from the example when it is missing.

| Variable | Default | Description |
| --- | --- | --- |
| `FASTAPI_HOST` | `127.0.0.1` | Backend bind host |
| `FASTAPI_PORT` | `9850` | Backend port |
| `UI_HOST` | `127.0.0.1` | Development server bind host |
| `UI_PORT` | `6422` | Development server port |
| `RELOAD` | `false` | Enable Uvicorn reload for development |
| `BACKEND_LOGS_VISIBLE` | `true` | Show backend logs in the launcher console |
| `NEURO_WEAVE_RESOURCES_DIR` | `data` | Resources and data directory |
| `NEURO_WEAVE_API_BASE_URL` | `/api` | API base URL |
| `NEURO_WEAVE_CORS_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Allowed CORS origins |

The local services bind to loopback by default. Set `FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, or `UI_PORT` to change their addresses. The default database and generated application data live under `data/`. Set `NEURO_WEAVE_RESOURCES_DIR` or pass `-ResourcesDir <path>` to use another resources directory; use the same location during installation and launch. Dataset registrations refer to external sources, so dataset bytes are not copied into the application.

## Documentation

- [Installation](assets/docs/runtime/installation.md) and [startup](assets/docs/runtime/startup.md)
- [Configuration](assets/docs/runtime/configuration.md) and [troubleshooting](assets/docs/runtime/troubleshooting.md)
- [User flows](assets/docs/product/user_flows.md)
- [Architecture overview](assets/docs/architecture/system_overview.md)
- [Contribution guide](assets/docs/development/contribution_guide.md)

## Limitations

- The supported launcher and runtime provisioning are Windows-only.
- Graph execution is limited to synthetic, forward-only validation; model training, checkpointing, prediction serving, dataset downloads, and preprocessing are not implemented.
- Exported weights are initialized and untrained.
- TensorFlow and JAX are CPU-only on native Windows; JAX CPU support is experimental and GPU execution requires a separate WSL2/Linux environment.
- PyTorch CUDA wheels are selected only when the detected NVIDIA driver supports them; otherwise the CPU wheel is used.
- Dataset registrations reference external sources; dataset bytes are not copied into the application.
- Trusted executable plugins and arbitrary custom code are not supported; custom nodes are template-based.

## Contributing

Contributions are welcome. Please read [assets/docs/development/contribution_guide.md](assets/docs/development/contribution_guide.md) and the coding standards under `assets/docs/development/` before opening a pull request. Keep changes focused and add or update tests where applicable.

## License

Licensed under the MIT License. See [LICENSE.md](LICENSE.md).