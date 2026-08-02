# Testing Strategy

Last updated: 2026-07-30

Backend unit tests cover manifest loading, DAG validation, shape propagation, parameter counts, and serialization. API integration tests cover save, reopen, revision updates, JSON import and export, and both code generators. Optional Keras binary export is skipped when dependencies are absent. Frontend tests cover graph-cycle logic and catalog filtering; critical canvas interactions should gain Playwright coverage next.
