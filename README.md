<h1>
  <img src="assets/brand/neuroweave-mark.svg" alt="" width="36" height="36" align="middle" />
  NeuroWeave Studio
</h1>

**Design, validate, save, and export machine-learning model graphs in a local visual workspace.**

[![License](https://img.shields.io/github/license/CTCycle/Neuro-Weave-Studio?style=flat-square)](LICENSE.md) [![Release](https://img.shields.io/github/v/release/CTCycle/Neuro-Weave-Studio?style=flat-square)](https://github.com/CTCycle/Neuro-Weave-Studio/releases) [![CI](https://img.shields.io/github/actions/workflow/status/CTCycle/Neuro-Weave-Studio/ci.yml?branch=develop&style=flat-square)](https://github.com/CTCycle/Neuro-Weave-Studio/actions/workflows/ci.yml?query=branch%3Adevelop) [![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white&style=flat-square)](app/server/pyproject.toml) [![Angular 22](https://img.shields.io/badge/Angular-22-DD0031?logo=angular&logoColor=white&style=flat-square)](app/client/package.json) [![Windows](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D4?logo=windows&logoColor=white&style=flat-square)](assets/docs/runtime/installation.md)

## Overview

NeuroWeave Studio is a locally hosted visual editor for designing and inspecting machine-learning model graphs. Its Angular interface and FastAPI service run on your computer, with project metadata stored in SQLite.

## Key Features

- Create model graphs from the node catalog, configure parameters, connect ports, and review graph and tensor-shape diagnostics.
- Save projects and graph revisions, and create reusable custom node templates.
- Register external datasets and their metadata without copying or downloading dataset contents.
- Run synthetic, forward-only functional validation and tensor inspection against an installed execution profile.
- Generate model source, project bundles, and supported model-format exports when the required runtime is available.

Graph execution is limited to synthetic, forward-only validation. Model training, checkpointing, prediction serving, dataset downloads, and preprocessing are not implemented. Exported weights are initialized and untrained.

## Technology Stack

- Angular 22 frontend and FastAPI backend with Python 3.13.
- SQLite persistence and Keras core dependencies.
- Optional PyTorch, TensorFlow, and JAX execution backends.
- The Windows launcher prepares pinned Python, uv, and Node.js runtimes in the repository.

## Installation and Usage

The supported quick-start launcher targets 64-bit Windows 10 or 11 and PowerShell 7. Git and internet access are needed for cloning and the initial runtime and dependency installation; separate Python and Node.js installations are not required.

```powershell
git clone https://github.com/CTCycle/Neuro-Weave-Studio.git
Set-Location .\Neuro-Weave-Studio
.\start_on_windows.ps1 -Action Install -InstallationType Standard
.\start_on_windows.ps1 -Action Launch
```

The first install creates `settings/.env` from [`settings/.env.example`](settings/.env.example), initializes or upgrades the SQLite database, and builds the frontend. The launcher starts the local services and opens the app in your default browser when possible. Default ports are `6422` for the frontend and `9850` for the backend.

To install development tools as well, use `-InstallationType Development`. Optional execution backends can be selected later through **Manage compute backends** in the launcher.

## Configuration

- Keep machine-specific values in the ignored `settings/.env` file; the launcher creates it from the example when it is missing.
- The local services bind to loopback by default. Set `FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, or `UI_PORT` to change their addresses.
- The default database and generated application data live under `data/`. Set `NEURO_WEAVE_RESOURCES_DIR` or pass `-ResourcesDir <path>` to use another resources directory; use the same location during installation and launch.
- Dataset registrations refer to external sources; dataset bytes are not copied into the application.

## Documentation

- [Installation](assets/docs/runtime/installation.md) and [startup](assets/docs/runtime/startup.md)
- [Configuration](assets/docs/runtime/configuration.md) and [troubleshooting](assets/docs/runtime/troubleshooting.md)
- [User flows](assets/docs/product/user_flows.md)

## License

Licensed under the MIT License. See [LICENSE.md](LICENSE.md).
