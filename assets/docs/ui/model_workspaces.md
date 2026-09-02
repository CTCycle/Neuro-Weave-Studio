# Model Catalog, Builder, and Workspaces

Last updated: 2026-09-02

The model catalog is the application home at `/models`. It is a library of persisted models backed by the `Project` aggregate and `/api/projects` endpoints. Catalog cards are summary-only: they show name, description, updated time, revision, restrained backend/API metadata, model input/output counts, node/edge counts, flat validation/compilation indicators, and whether source exists for the current revision. Input counts use manifest semantic roles; output counts use the same explicit-or-inferred effective-output rules as validation. Search and compiled/validation segmented filters operate on that in-memory summary list. JSON project import and `.keras` architecture import are explicit catalog actions.

## Responsibilities

- `/builder` creates a model and `/builder/:modelId` edits its architecture. The Builder owns graph authoring, node configuration, validation, undo/redo, naming, description, the persisted execution profile, and explicit save. It exposes Save, Validate, undo, and redo. Compilation settings, source generation, inference, export, duplicate, import/export, and model browsing belong to dedicated workspace flows.
- `/models/:modelId` is a read-only overview with a Foblex-backed architecture preview, configuration/readiness summaries, management actions, and links to dedicated workflows.
- `/models/:modelId/compilation` owns capability-driven Keras/native configuration. Each effective graph output is shown explicitly; loss and metrics are configured together per output, using stable `node_id:port_id` selectors for multi-output models and `__default__` for the single-output case. Task metadata may preselect compatible catalogue options, but backend validation remains authoritative. Blocking findings prevent an invalid compilation configuration from being persisted. The page then generates and previews a source snapshot for the exact saved revision.
- `/models/:modelId/export` consumes the server-owned export capability matrix and owns project/source downloads plus verified binary export jobs with runtime/API/device/policy diagnostics and a gated artifact download.
- `/models/:modelId/inference` remains directly routable for isolated saved-graph smoke jobs, but it is no longer a primary workspace tab. Navigation foregrounds Overview, Compilation, and Export. The inference route still polls lifecycle progress, exposes Stop, and retains the explicit no-training/no-weight-updates message.
- Training remains a disabled planned card only; it has no route, schema, or backend contract.

The selected model workspace header shows input/output counts alongside node/connection totals so model boundaries are visible without opening the Builder. Workspace management actions are deliberately explicit: the Export workflow downloads the persisted JSON bundle, duplicate opens the new model workspace, and deletion requires an in-app confirmation dialog.

Catalog/workspace data is separate from the editable Angular graph store. The store is initialized only for Builder routes, while workspace previews use the same node visuals and drafting-board styling in a read-only Foblex canvas with a scoped store. `BuilderTabsStore` holds independent sessions and saved baselines for each open tab; navigation is guarded when any tab is dirty, and users can stay or discard changes before leaving.

The Environment page owns session-scoped API, backend, device, and memory-policy selection plus active-worker diagnostics. The application shell keeps Models and Model Builder in the primary navigation and reuses the existing cards, notices, drawers, controls, typography, and responsive conventions. Below 1024px, Builder retains the compact Catalog/Inspector drawer behavior rather than introducing a second visual language.
