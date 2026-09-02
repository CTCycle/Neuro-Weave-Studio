# Shape Inference

Last updated: 2026-09-02

Shape inference is manifest-driven and runs through the backend validation pipeline before compilation or execution. Tensor specifications retain symbolic or unknown dimensions where the graph cannot resolve them statically, and downstream rules validate rank and dimension compatibility without requiring runtime framework imports.

Model input nodes persist feature dimensions only in graph schema 5. The runtime batch dimension is synthesized as a leading unknown dimension when validation, exporters, or workers consume an input shape. A persisted tabular shape `[20]` therefore participates in inference as `[None, 20]`; an image shape `[224, 224, 3]` participates as `[None, 224, 224, 3]`. The Builder Inspector hides that runtime axis by default and can show it as a read-only preview. Toggling the preview never changes the saved graph.

`input.parameter` is not a runtime sample input and keeps its declared shape unchanged. Keras architecture import follows the same boundary: `batch_shape` and `batch_input_shape` lose their leading batch element, while `input_shape` is retained as feature-only. The database migration `20260902_runtime_batch_shapes` performs the equivalent one-time normalization for persisted projects and revision snapshots.

Validation returns per-node `ShapeSummary` values, which the Angular Builder uses for port inspection and model diagnostics. Generated source and isolated workers continue to receive batch-inclusive runtime specifications derived from the canonical feature-only graph.
