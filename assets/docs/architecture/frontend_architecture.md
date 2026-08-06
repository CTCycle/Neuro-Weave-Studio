# Frontend Architecture

Last updated: 2026-08-05

The canonical React application lives in `app/client/src`. Route-level orchestration lives in `App.tsx`. Typed API access is isolated in `api/client.ts` and targets the relative `VITE_API_BASE_URL` path. Editor domain types are in `domain/types.ts`. Zustand owns graph state, history, selection, configuration, and serialization. XYFlow renders the canvas. Reusable panels implement catalog browsing, properties, validation, code preview, and project commands. Node rendering is registry-based through XYFlow node types and backend manifests.

`/api/metadata` supplies the four `code_targets` and operational `execution_backends` with reason codes. Keras 3 is shown as a fixed model-API badge; code target is persisted per project, while execution backend is session/request scoped. None, one, and multiple operational backends map to disabled, auto-selected, and explicit-choice states respectively.

The workspace preserves the drafting-board palette while reflowing at 1440/1200/1024px breakpoints. At 1024px and above graph authoring remains fully supported; smaller viewports use non-clipping Catalog/Inspector drawers and a recommended-size notice.
