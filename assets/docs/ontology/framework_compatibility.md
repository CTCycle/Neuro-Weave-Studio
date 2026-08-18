# Framework Compatibility

Last updated: 2026-07-30

A node definition contains declarative mappings for the v2 code targets `keras`, `tensorflow`, `pytorch`, and `jax`. Keras 3 is the model API; the latter three are native ecosystem targets. The checked-in manifest is the source of target and parameter capability truth; the loader must not promote nodes into a portable set. Unsupported node/parameter combinations produce typed compatibility diagnostics and block compilation—there is no identity fallback.
