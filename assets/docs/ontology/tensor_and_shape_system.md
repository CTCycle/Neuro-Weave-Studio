# Tensor and Shape System

Last updated: 2026-09-02

Tensor Specification contains an ordered shape, data type, and optional name. A dimension is a positive integer, an unknown value, or a symbolic dimension. Rank is the number of dimensions. Tensor compatibility requires matching dtypes and ranks; unknown dimensions remain compatible unless a concrete contradiction is known. Shape inference never treats `-1` as a concrete positive dimension in persisted canonical output.

NeuroWeave distinguishes persisted feature shapes from runtime tensor shapes:

- Persisted graph input parameters contain only features, for example `[8]`, `[12, 4]`, or `[224, 224, 3]`.
- Runtime input parameters expose `[None, ...features]` to validation, model construction, and inference. The injected batch dimension is never written back to a graph, revision, export bundle, or database row.
- The Inspector's “Show batch dimension” control is presentation-only. It previews the runtime form and does not change saved node parameters.

Keras import follows the same contract. `input_shape` is already feature-only. For `batch_shape` and `batch_input_shape`, the first value is removed whether it is `None` or a concrete batch size; a concrete value is reported as a warning because batch sizing is runtime policy. The resulting feature rank selects tensor (rank 1), sequence (rank 2), or image (rank 3 or greater) input manifests. Missing, empty, non-positive, or otherwise malformed feature shapes are rejected instead of being double-batched.

Quantization configuration is optional. Empty objects, including recursively empty placeholder objects, mean disabled configuration; nonempty nested values and meaningful scalar values such as `false` or `0` are preserved.
