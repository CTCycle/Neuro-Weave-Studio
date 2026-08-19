# User Flows

Last updated: 2026-08-20

The application opens on the model catalog (`/models`). Users browse or filter saved models, select one for a read-only overview, and choose a dedicated Compilation or Inference workspace. The Model Builder (`/builder`) is reserved for creating a model; `/builder/:modelId` edits an existing architecture. In Builder, users select a code target, drag nodes from the catalog, connect typed ports, configure every exposed node property, validate, and explicitly save. Save and Save & compile persist the draft; the latter also creates source for the current revision. Unsaved Builder changes warn before internal navigation or browser unload.

The model overview supports architecture inspection, duplicate, JSON export, and confirmed delete. Training is represented as a disabled planned capability rather than a route or backend contract. Reopening restores the graph and revision. Compilation source snapshots are revision-specific, while binary Keras export remains a separate runtime-backend choice. Inference is a saved-graph smoke test and explicitly performs no training or weight updates.

For binary Keras export and inference, the Environment page discovers live API/backend/device/policy capability from `/api/metadata`. It disables unavailable native capability, shows supported memory policies per device, and keeps selection session-scoped. Each operation is a job: progress, PID, device, RSS, exit, cancellation, and failure diagnostics remain visible; the artifact download is enabled only after verified success. If an active job uses the current selection, users choose Stop and switch or Keep running/use the new selection later. Keras remains the model API layer; native inference and training are deferred.
