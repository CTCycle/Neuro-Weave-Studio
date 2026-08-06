# Framework Compatibility

Last updated: 2026-07-30

A node definition contains mappings for the v2 code targets `keras`, `tensorflow`, `pytorch`, and `jax`. Keras 3 is the model API; the latter three are native ecosystem targets. Compatibility is evaluated per node and whole graph through an explicit adapter registry. Unsupported node/parameter combinations produce typed compatibility diagnostics and block compilation—there is no identity fallback.
