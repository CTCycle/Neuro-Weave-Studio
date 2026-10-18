# NeuroWeave Studio

Last updated: 2026-10-18

NeuroWeave Studio is a local-first visual editor for building, validating, saving, and exporting machine-learning model graphs. Version 1.0.0 uses a framework-independent graph, manifest-defined nodes, a FastAPI backend, and an Angular/Foblex frontend.

## Current scope

- The Builder edits typed graphs and validates topology, parameters, tensor shapes, and the selected execution profile.
- Functional Validation and Tensor Inspection run synthetic forward-only checks. They do not train models, persist weight updates, or serve predictions.
- Each project stores one backend and either the Keras API or a verified native implementation. Unsupported mappings and unavailable runtimes fail closed; native paths do not fall back to Keras.
- Source and project downloads are available alongside capability-gated binary exports. Exported models are initialized and untrained; Keras architecture import does not import weights or custom code.
- Dataset registrations store definitions and bounded metadata for external sources. Processing and split settings are declarative; partitions and training data are not materialized.

## Run on Windows

~~~powershell
.\start_on_windows.ps1
~~~

The launcher checks runtime readiness and starts the local app. Use the [installation](assets/docs/runtime/installation.md), [startup](assets/docs/runtime/startup.md), and [configuration](assets/docs/runtime/configuration.md) guides for setup, data paths, and launcher options.

## Documentation

Start at the [project documentation index](assets/docs/project_index.md). The [status ledger](assets/docs/project_status_ledger.md) summarizes revision-scoped validation results and open boundaries.

## License

MIT. See [LICENSE.md](LICENSE.md).
