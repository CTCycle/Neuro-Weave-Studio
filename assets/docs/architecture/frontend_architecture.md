# Frontend Architecture

Last updated: 2026-08-07

The canonical React application lives in `app/client/src`. Route-level orchestration lives in `App.tsx`, with `AppShell` providing primary navigation for the model catalog and Model Builder. Typed API access is isolated in `api/client.ts` and targets the relative `VITE_API_BASE_URL` path. Editor domain types are in `domain/types.ts`.

The model catalog and model workspaces are reached through `/models`, `/models/:modelId`, `/models/:modelId/compilation`, and `/models/:modelId/inference`. New and existing architecture authoring is isolated under `/builder` and `/builder/:modelId`. Catalog/workspace data is held by the application resource context; Zustand is loaded only by Builder routes and owns graph state, history, selection, configuration, and serialization. This prevents browsing or inspecting a model from mutating the editable draft. XYFlow renders both the editable canvas and a read-only workspace preview. Reusable panels implement catalog browsing, properties, validation, code preview, smoke inference, and model commands. Node rendering is registry-based through XYFlow node types and backend manifests.

`/api/metadata` supplies the four `code_targets` and operational `execution_backends` with reason codes. Keras 3 is shown as a fixed model-API badge; code target is persisted per project, while execution backend is session/request scoped. None, one, and multiple operational backends map to disabled, auto-selected, and explicit-choice states respectively.

The workspace preserves the drafting-board palette while reflowing at 1440/1200/1024px breakpoints. At 1024px and above graph authoring remains fully supported; smaller viewports use non-clipping Catalog/Inspector drawers and a recommended-size notice.
