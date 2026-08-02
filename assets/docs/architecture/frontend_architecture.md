# Frontend Architecture

Last updated: 2026-08-02

React route-level orchestration lives in `App.tsx`. Typed API access is isolated
in `api/client.ts`, and editor contracts are in `domain/types.ts`. Zustand owns
the graph, model configuration, selected node, undo/redo history, and local
serialization state. XYFlow renders nodes, typed handles, edges, pan/zoom,
minimap, and selection.

The client workflow is deliberately layered:

1. `NodeCatalog` filters backend manifests and starts node creation.
2. `GraphCanvas` renders and edits the graph while applying immediate client
   cycle/port feedback.
3. `PropertiesInspector` and `ModelConfigurationPanel` edit typed parameters.
4. `ValidationPanel` displays backend diagnostics, tensor shapes, and counts.
5. `CodePreview` and `ProjectToolbar` invoke compilation, save, reopen,
   duplicate, import, and export commands.

The UI does not treat local state as persisted truth. Save responses replace the
current document and revision, and HTTP 409 conflicts require reopening before
overwriting a newer server revision.
