# User Flows

Last updated: 2026-08-07

The application opens on the model catalog (`/models`). Users browse or filter saved models, select one for a read-only overview, and choose a dedicated Compilation or Inference workspace. The Model Builder (`/builder`) is reserved for creating a model; `/builder/:modelId` edits an existing architecture. In Builder, users select a code target, drag nodes from the catalog, connect typed ports, configure every exposed node property, validate, and explicitly save. Save and Save & compile persist the draft; the latter also creates source for the current revision. Unsaved Builder changes warn before internal navigation or browser unload.

The model overview supports architecture inspection, duplicate, JSON export, and confirmed delete. Training is represented as a disabled planned capability rather than a route or backend contract. Reopening restores the graph and revision. Compilation source snapshots are revision-specific, while binary Keras export remains a separate runtime-backend choice. Inference is a saved-graph smoke test and explicitly performs no training or weight updates.

For binary Keras export, the editor discovers operational TensorFlow, PyTorch, and JAX backends from `/api/metadata`. It disables export when none are available, auto-selects the only available backend, and requires an explicit selector choice when several are available. This is an export-runtime choice and is never stored in project data. Keras remains the model API layer; the selected backend is the runtime used by the isolated worker.
