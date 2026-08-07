# Model Catalog, Builder, and Workspaces

Last updated: 2026-08-07

The model catalog is the application home at `/models`. It is a library of persisted models backed by the existing `Project` aggregate and `/api/projects` endpoints. Catalog cards are summary-only: they show name, description, updated time, revision, code target, node/edge counts, validation state, and whether the configured target has a source snapshot for the current revision. Search and compiled/validation filters operate on that in-memory summary list.

## Responsibilities

- `/builder` creates a model and `/builder/:modelId` edits its architecture. The Builder owns graph authoring, node configuration, model defaults, validation, undo/redo, naming, description, code target, and explicit save. It exposes Save, Save & compile, Validate, undo, and redo. Usage controls such as execution-backend selection, smoke inference, binary export, duplicate, import/export, and model browsing belong to the workspace.
- `/models/:modelId` is a read-only overview with an XYFlow architecture preview, configuration/readiness summaries, management actions, and links to dedicated workflows.
- `/models/:modelId/compilation` persists a source snapshot for the configured target and current revision, exposes generated source viewing/copying, and keeps `.keras` binary export separate with its runtime-backend availability selector.
- `/models/:modelId/inference` contains the saved-graph smoke test and retains the explicit no-training/no-weight-updates message.
- Training remains a disabled planned card only; it has no route, schema, or backend contract.

Workspace management actions are deliberately explicit: JSON export downloads the persisted bundle, duplicate opens the new model workspace, and deletion requires an in-app confirmation dialog. Generated-source copy reports success or clipboard unavailability and includes a document fallback for browser contexts without the Clipboard API.

Catalog/workspace data is separate from the editable Zustand graph store. The store is initialized only for Builder routes, while workspace previews use the existing node visuals and drafting-board styling in a read-only XYFlow canvas. Builder navigation is guarded by a baseline of the last loaded or saved graph and metadata; users can stay or discard changes before leaving.

The application shell keeps Models and Model Builder in the primary navigation and reuses the existing cards, notices, drawers, controls, typography, and responsive conventions. Below 1024px, Builder retains the compact Catalog/Inspector drawer behavior rather than introducing a second visual language.
