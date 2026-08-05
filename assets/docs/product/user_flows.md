# User Flows

Last updated: 2026-08-05

The main flow is: create or open a project, select a framework target, drag nodes from the catalog, connect typed ports, configure node properties, validate, review shape and parameter summaries, preview generated code, then save or export JSON. Reopening restores the graph and revision. Duplicate creates an independent project.

For binary Keras export, the editor discovers operational TensorFlow/Torch backends from `/api/metadata`. It disables export when none are available, auto-selects the only available backend, and requires an explicit selector choice when several are available. This is an export-runtime choice and is never stored in project data. Keras remains the model API layer; the selected backend is the runtime used by the isolated worker.
