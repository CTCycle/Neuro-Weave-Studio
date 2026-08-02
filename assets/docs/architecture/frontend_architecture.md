# Frontend Architecture

Last updated: 2026-07-30

React route-level orchestration lives in `App.tsx`. Typed API access is isolated in `api/client.ts`. Editor domain types are in `domain/types.ts`. Zustand owns graph state, history, selection, configuration, and serialization. XYFlow renders the canvas. Reusable panels implement catalog browsing, properties, validation, code preview, and project commands. Node rendering is registry-based through XYFlow node types and backend manifests.
