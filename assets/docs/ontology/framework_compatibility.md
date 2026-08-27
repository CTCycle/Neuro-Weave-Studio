# Framework Compatibility

Last updated: 2026-08-27

A node definition contains declarative implementation mappings keyed by `keras`, `tensorflow`, `pytorch`, and `jax`. The model execution profile resolves to `keras` when `keras_enabled=true`, or to the persisted backend when it is false. The checked-in manifest is the source of implementation and parameter capability truth; the loader must not promote nodes into a portable set. Native TensorFlow, PyTorch, and JAX/Flax runtime/source support is restricted to the verified common subset. The native compiler controls are separate live capabilities, not assumptions about manifest support. Unsupported node/parameter combinations produce typed compatibility diagnostics and block compilation; there is no identity fallback.

Runtime/API and native-compiler availability is live probe data returned by `/api/metadata`. Missing optional dependencies or an unavailable compiler disable only the affected capability with an explicit reason; source generation remains available where the manifest/exporter contract is valid. The validation and node catalog surfaces keep Keras portability, native backend support, compiler availability, and runtime execution as separate concerns.
