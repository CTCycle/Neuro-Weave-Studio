# ADR 002: Graph Library

Last updated: 2026-08-31

Status: accepted; this decision supersedes the original XYFlow selection. Foblex Flow 19.1.6 is the Angular canvas adapter. It provides node/connection rendering, custom connectors, pan/zoom, minimap, snapping, and drag events while the application retains ownership of graph records, connection validation, history, layout, and persistence. The adapter boundary in `features/builder/graph/foblex-graph.adapter.ts` prevents library-specific records from becoming the model contract. A separate dataflow engine would duplicate backend compilation semantics; Three.js remains unsuitable because the editor is two-dimensional.
