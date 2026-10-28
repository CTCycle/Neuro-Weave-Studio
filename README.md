<p align="center">
  <img src="assets/brand/neuroweave-mark.svg" alt="NeuroWeave Studio" width="88" />
</p>

<h1 align="center">NeuroWeave Studio</h1>

<p align="center">
  Design and inspect machine-learning model graphs in a local visual workspace.
</p>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/Angular-22-DD0031?logo=angular&logoColor=white" alt="Angular 22" />
  <img src="https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D4?logo=windows&logoColor=white" alt="Windows 10 or 11" />
</p>

NeuroWeave Studio runs an Angular interface and FastAPI service on your computer. Build a model as a directed graph, inspect its tensor shapes, save revisions, and generate source or supported exports. Projects are stored in a local SQLite database.

## What you can do

- Add layers from a searchable node catalog, configure their parameters, and connect typed ports in the Model Builder.
- Save model projects and graph revisions, then review graph and tensor-shape diagnostics.
- Create reusable mathematical and sequential node templates.
- Register datasets by referencing their existing files. The app stores definitions and inspection metadata; it does not copy dataset contents.
- Generate Keras source, project bundles, and supported model-format exports when the required runtime is installed.

Functional validation runs a synthetic forward pass with initialized weights. The application does not train models or serve predictions; binary exports are untrained.

## Screenshots

The Compilation workspace shows a validated execution profile and source snapshot for a saved revision:

![Compilation workspace showing a validated Keras and PyTorch profile with generated Python source](assets/QA/readme-20261028/compilation.png)

## Quick start

The supported launcher runs on 64-bit Windows 10 or 11 and requires PowerShell 7. Git and internet access are needed to clone the repository and install its pinned runtimes. You do not need to install Python or Node.js separately.

```powershell
git clone https://github.com/CTCycle/Neuro-Weave-Studio.git
Set-Location .\Neuro-Weave-Studio
.\start_on_windows.ps1 -Action Install -InstallationType Standard
.\start_on_windows.ps1 -Action Launch
```

The launcher installs the base environment, initializes the database, builds the frontend, and opens `http://127.0.0.1:9850/`. On first launch, choose an execution backend if you plan to run functional validation or binary exports. Keras-only mode supports graph editing and source generation.

To start the frontend and backend as separate development processes, follow the [development startup guide](assets/docs/runtime/startup.md). See the [installation guide](assets/docs/runtime/installation.md) for backend options and runtime details.

## Typical workflow

1. Choose **New model** and add nodes from the catalog.
2. Connect ports, set input shapes, and configure node parameters.
3. Save the model to persist its graph and revision.
4. Open **Functional Validation** to run a synthetic forward pass with an installed execution profile.
5. Use **Compilation** to inspect generated source or **Export** to create a supported artifact.

## Technology and platform

The frontend uses Angular 22 and TypeScript. The backend uses FastAPI on Python 3.13, with SQLite for persistence and Keras 3 as the model API. PyTorch, TensorFlow, and JAX are optional execution backends. The supported Windows launcher provisions pinned Python, uv, and Node.js versions under `runtimes/`.

The launcher is Windows-only. Native Windows backend capabilities vary: TensorFlow and JAX are CPU-only, JAX CPU support is experimental, and PyTorch CUDA depends on the installed NVIDIA driver. GPU execution for TensorFlow and JAX requires a separate WSL2/Linux environment.

## Documentation

- [Installation and backend setup](assets/docs/runtime/installation.md)
- [Startup and development workflow](assets/docs/runtime/startup.md)
- [Configuration](assets/docs/runtime/configuration.md)
- [User flows](assets/docs/product/user_flows.md)
- [Architecture overview](assets/docs/architecture/system_overview.md)
- [Contribution guide](assets/docs/development/contribution_guide.md)

## Contributing

Please read the [contribution guide](assets/docs/development/contribution_guide.md) and the relevant coding standards under `assets/docs/development/` before opening a pull request.

## License

NeuroWeave Studio is licensed under the [MIT License](LICENSE.md).
