# Getting Started

Last updated: 2026-08-02

NeuroWeave is a local-first editor for designing machine-learning architectures
without writing the graph by hand. It does not train models or manage learned
weights; it produces validated architecture definitions and source artifacts.

## Start the application

On Windows, use:

```powershell
.\start_on_windows.bat -Mode Launch
```

The launcher starts the backend at `http://127.0.0.1:8000` and the frontend at
`http://127.0.0.1:5173`. Use `-NoBrowser` in automation. The first backend start
creates the default SQLite database when it is missing.

For source development, run `-Mode Dev`. Use `-Mode Stop` to stop only the
processes recorded by the launcher. See [runtime startup](../runtime/startup.md)
for manual commands and maintenance modes.

## First project

1. Choose **New Project** in the toolbar.
2. Select a target framework.
3. Drag an input node from the catalog onto the canvas.
4. Add layers and an output node, then connect compatible ports.
5. Configure node parameters and model-level training metadata.
6. Run validation and resolve blocking diagnostics.
7. Review tensor shapes, parameter counts, and generated source.
8. Save the project.

The backend validation result is authoritative. A graph that looks connected in
the canvas can still be rejected for cycles, missing required inputs, invalid
parameters, shape incompatibility, or unsupported framework mappings.

## Data and privacy

Projects are stored locally in the configured SQLite or PostgreSQL database.
Generated exports are written below the configured export root. Do not treat the
MVP as a training-data store or enter secrets into project descriptions or graph
parameters.
