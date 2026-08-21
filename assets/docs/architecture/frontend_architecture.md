# Frontend Architecture

Last updated: 2026-08-21

The canonical React application lives in `app/client/src`. Route-level orchestration lives in `App.tsx`, with `AppShell` providing primary navigation for the model catalog and Model Builder. Typed API access is isolated in `api/client.ts` and targets the relative `VITE_API_BASE_URL` path. Editor domain types are in `domain/types.ts`.

The model catalog and model workspaces are reached through `/models`, `/models/:modelId`, `/models/:modelId/compilation`, and `/models/:modelId/inference`. New and existing architecture authoring is isolated under `/builder` and `/builder/:modelId`. Catalog/workspace data is held by the application resource context; Builder uses Zustand for graph state, history, selection, configuration, and serialization, with `builderTabsStore` retaining a separate graph session and saved baseline per open model tab. This prevents browsing or switching models from mutating another editable draft. XYFlow renders both the editable canvas and a read-only workspace preview. Reusable panels implement catalog browsing, properties, validation, code preview, smoke inference, and model commands. Node rendering is registry-based through XYFlow node types and manifest implementation mappings.

`/api/metadata` supplies live `execution_backends`, runtime API status, and the metadata-driven compilation catalog. A model's `execution_backend` and `keras_enabled` are persisted together; the Builder exposes them as backend and Keras controls. Backend availability informs capability badges and runtime preferences, but it does not replace the persisted model profile.

The workspace preserves the drafting-board palette while reflowing at 1440/1200/1024px breakpoints. Its fixed-height editor uses a tab bar, authoring toolbar, node catalog, compact compilation summary, XYFlow canvas, inspector, and a bottom validation region with a pointer-resizable splitter. At 1024px and below the Catalog and Inspector become explicit drawers; the canvas and validation region remain within the viewport without document scrolling.
