# NeuroWeave Studio

[![Release](https://img.shields.io/badge/release-v1.0.0-2ea44f)](https://github.com/CTCycle/Neuro-Weave-Studio/releases/tag/v1.0.0)
[![Python](https://img.shields.io/badge/python-3.13.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/node.js-22.23.1-339933?logo=node.js)](https://nodejs.org/)
[![License](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE.md)
[![CI](https://github.com/CTCycle/Neuro-Weave-Studio/actions/workflows/ci.yml/badge.svg)](https://github.com/CTCycle/Neuro-Weave-Studio/actions/workflows/ci.yml)

Last updated: 2026-10-18

NeuroWeave Studio is a locally hosted visual editor for designing, validating,
saving, and exporting machine-learning model graphs. The Angular interface and
FastAPI service run on your computer, with project metadata stored in SQLite.
The supported quick-start launcher currently targets Windows 10 and 11.

> **Scope:** Functional Validation and Tensor Inspection run synthetic,
> forward-only checks. They do not train models or establish predictive
> quality. Exported model weights are initialized and untrained.

## Main workflows

1. **Design a model:** choose nodes from the manifest-driven catalog, configure
   parameters, connect ports, and review graph, shape, and execution-profile
   validation in the Builder.
2. **Save and organize work:** create projects, keep graph revisions in the
   local workspace, and save reusable custom node templates.
3. **Register datasets:** record external dataset sources and metadata for a
   project. Dataset bytes stay at their source; registration does not download
   data or materialize preprocessing and split partitions.
4. **Inspect and validate:** run Tensor Inspection or Functional Validation
   with synthetic inputs against an available execution profile.
5. **Export:** generate source, a project bundle, or a supported model-format
   artifact when its required runtime is available. Export checks the artifact
   before publishing it.

Training, optimizer updates, checkpointing, prediction serving, dataset
downloads, and preprocessing are outside v1.0.0. See the [changelog](CHANGELOG.md)
for the initial stable release scope.

## Start NeuroWeave

### Prerequisites

- Windows 10 or 11 with PowerShell 7.
- Git to clone the repository.
- Internet access for the initial runtime and dependency installation.

The launcher installs the repository-pinned Python 3.13.14, uv 0.12.0, and
Node.js 22.23.1 under `runtimes/`. You do not need to install those runtimes
separately. Normal launches reuse valid dependencies and frontend build output.

### Install and launch

Open PowerShell 7 and run:

```powershell
git clone https://github.com/CTCycle/Neuro-Weave-Studio.git
Set-Location .\Neuro-Weave-Studio
.\start_on_windows.ps1 -Action Install -InstallationType Standard
.\start_on_windows.ps1 -Action Launch
```

On first installation, the launcher creates `settings/.env` from the tracked
[`settings/.env.example`](settings/.env.example). It installs the pinned
runtimes and locked dependencies, initializes or upgrades the SQLite database,
and builds the frontend. It then starts the backend and frontend and prints
their local URLs and process IDs. Open the frontend URL in your browser. The
default ports are `6422` for the frontend and `9850` for the backend.

To install the development tools (Ruff, Pyright, and pytest) as well, use the
Development installation profile:

```powershell
.\start_on_windows.ps1 -Action Install -InstallationType Development
```

Keras is part of the base backend environment. Optional execution backends can
be installed when needed; for example, to add PyTorch CPU support:

```powershell
.\start_on_windows.ps1 -Action ManageBackends -Backends pytorch -Acceleration CPU
```

Functional Validation and model-format exports require a compatible installed
backend and an available execution profile. Source generation and project
editing can be used without an optional execution backend.

### Stop the local services

Use the backend and frontend process IDs printed by the current launch:

```powershell
taskkill.exe /PID <backend-pid> /T /F
taskkill.exe /PID <frontend-pid> /T /F
```

Replace each placeholder with the matching ID from that launch. Confirm that
the configured ports are free before starting another instance.

## Configuration and local data

Local use does not require credentials. Keep machine-specific settings in the
ignored `settings/.env` file; do not commit it. If missing, the launcher copies
the credential-free [`settings/.env.example`](settings/.env.example) template
and leaves existing settings unchanged.

The default services bind to loopback and are available only from the local
computer. Set `FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, and `UI_PORT` in
`settings/.env` to change the addresses. Set `NEURO_WEAVE_RESOURCES_DIR` to
move the database and generated application data together, or pass
`-ResourcesDir <path>` to launcher actions. Use the same resources directory
for installation and launch.

| Data | Default location |
| --- | --- |
| Local configuration | `settings/.env` |
| SQLite database | `data/database.db` |
| Exports | `data/runtime/exports` |
| Temporary job workspaces and logs | `data/runtime/ml-jobs` and `data/runtime/logs` |
| Downloaded runtimes, dependencies, and build output | `runtimes/` |

Registered dataset definitions refer to external sources; NeuroWeave does not
copy their contents. Back up the database and exported artifacts you want to
keep before removing local application data. See
[runtime configuration](assets/docs/runtime/configuration.md) for per-path
overrides and additional environment variables.

## Troubleshooting

- **A port is already in use:** change the relevant port in `settings/.env`, or
  close the listener only after confirming it belongs to your previous app
  launch. The launcher reports the listener before asking to stop it.
- **The frontend build is missing or stale:** from the repository root, run
  `.\start_on_windows.ps1 -Action RebuildFrontend`.
- **An execution profile is unavailable:** install its optional backend with
  `ManageBackends`, then refresh capability discovery in the Environment page.
- **Database migration fails:** preserve and back up the database, then follow
  [runtime troubleshooting](assets/docs/runtime/troubleshooting.md). Do not
  delete the database as a first repair step.
- **PyTorch native compilation is unavailable:** `torch.compile` on Windows
  requires the Visual Studio C++ workload and Windows SDK. Ordinary eager
  validation remains available when that toolchain is absent.

## Documentation

- [Documentation index](assets/docs/project_index.md) — project documentation
  map, architecture, behavior contracts, and current status.
- [User flows](assets/docs/product/user_flows.md) — catalog, Builder, workspace,
  dataset, validation, and export journeys.
- [Installation](assets/docs/runtime/installation.md) and
  [startup](assets/docs/runtime/startup.md) — launcher behavior and local
  runtime setup.
- [Configuration](assets/docs/runtime/configuration.md) and
  [troubleshooting](assets/docs/runtime/troubleshooting.md) — local paths,
  environment settings, migrations, and common runtime problems.
- [Testing strategy](assets/docs/development/testing_strategy.md) — checks and
  the limits of their evidence.
- [Changelog](CHANGELOG.md) — v1.0.0 features and scope.

## License

MIT. See [LICENSE.md](LICENSE.md).
