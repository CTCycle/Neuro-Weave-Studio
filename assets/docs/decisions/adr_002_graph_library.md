# ADR 002: Graph Library

Last updated: 2026-07-30

Status: accepted. XYFlow is selected. It provides node and edge rendering, custom handles, connection validation, minimap, controls, subflows, pan and zoom, and a mature React API. Rete.js offers a stronger plugin and dataflow-engine abstraction, but NeuroWeave already owns compilation and execution semantics in the backend, so adopting another engine would duplicate domain responsibility. Three.js is rejected because the editor is two-dimensional.
