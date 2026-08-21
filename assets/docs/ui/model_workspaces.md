# Model Catalog, Builder, and Workspaces

Last updated: 2026-08-21

The model catalog is the application home at `/models`. It is a library of persisted models backed by the `Project` aggregate and `/api/projects` endpoints. Catalog cards are summary-only: they show name, description, updated time, revision, execution backend, Keras/native mode, node/edge counts, validation state, and whether source exists for the current revision. Search and compiled/validation filters operate on that in-memory summary list. JSON project import and `.keras` architecture import are explicit catalog actions.

## Responsibilities

- `/builder` creates a model and `/builder/:modelId` edits its architecture. The Builder owns graph authoring, node configuration, model defaults, validation, undo/redo, naming, description, the persisted execution profile, and explicit save. It exposes Save, Save & compile, Validate, undo, and redo. Usage controls such as smoke inference, binary export, duplicate, import/export, and model browsing belong to the workspace.
- `/models/:modelId` is a read-only overview with an XYFlow architecture preview, configuration/readiness summaries, management actions, and links to dedicated workflows.
- `/models/:modelId/compilation` persists a source snapshot for the execution profile and current revision, exposes generated source viewing/copying/downloading, and submits `.keras` binary export jobs with runtime/API/device/policy diagnostics and a gated download.
- `/models/:modelId/inference` submits an isolated saved-graph smoke job, polls lifecycle progress, exposes Stop, and retains the explicit no-training/no-weight-updates message. Navigation does not silently cancel server work.
- Training remains a disabled planned card only; it has no route, schema, or backend contract.

Workspace management actions are deliberately explicit: JSON export downloads the persisted bundle, duplicate opens the new model workspace, and deletion requires an in-app confirmation dialog. Generated-source copy reports success or clipboard unavailability and includes a document fallback for browser contexts without the Clipboard API.

Catalog/workspace data is separate from the editable Zustand graph store. The store is initialized only for Builder routes, while workspace previews use the existing node visuals and drafting-board styling in a read-only XYFlow canvas. `builderTabsStore` holds independent sessions and saved baselines for each open tab; navigation is guarded when any tab is dirty, and users can stay or discard changes before leaving.

The Environment page owns session-scoped API, backend, device, and memory-policy selection plus active-worker diagnostics. When a selection is used by a live job it offers Stop and switch or Keep running/use for later jobs. The application shell keeps Models and Model Builder in the primary navigation and reuses the existing cards, notices, drawers, controls, typography, and responsive conventions. Below 1024px, Builder retains the compact Catalog/Inspector drawer behavior rather than introducing a second visual language.
