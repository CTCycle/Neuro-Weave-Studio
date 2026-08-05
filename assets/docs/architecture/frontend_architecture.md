# Frontend Architecture

Last updated: 2026-08-05

The canonical React application lives in `app/client/src`. Route-level orchestration lives in `App.tsx`. Typed API access is isolated in `api/client.ts` and targets the relative `VITE_API_BASE_URL` path. Editor domain types are in `domain/types.ts`. Zustand owns graph state, history, selection, configuration, and serialization. XYFlow renders the canvas. Reusable panels implement catalog browsing, properties, validation, code preview, and project commands. Node rendering is registry-based through XYFlow node types and backend manifests.

`/api/metadata` supplies operational Keras backends for the toolbar selector. The selector is disabled with an installation hint when none are available, auto-selected when exactly one is available, and left unselected when several require an explicit export-time choice. It is not part of persisted project state.
