# Shape Inference

Last updated: 2026-10-21

Shape inference is manifest-driven and runs through the backend validation pipeline before compilation or execution. Tensor specifications retain symbolic or unknown dimensions where the graph cannot resolve them statically, and downstream rules validate rank and dimension compatibility without requiring runtime framework imports.

Model input nodes persist feature dimensions only in graph schema 6. The runtime batch dimension is synthesized as a leading unknown dimension when validation, exporters, or workers consume an input shape. A persisted tabular shape `[20]` therefore participates in inference as `[None, 20]`; an image shape `[224, 224, 3]` participates as `[None, 224, 224, 3]`. The Builder Inspector hides that runtime axis by default and can show it as a read-only preview. Toggling the preview never changes the saved graph.

`input.parameter` is not a runtime sample input and keeps its declared shape unchanged. Keras architecture import follows the same boundary: `batch_shape` and `batch_input_shape` lose their leading batch element, while `input_shape` is retained as feature-only. Database startup does not transform historical project shapes; databases using removed schema revisions must be recreated explicitly.

Validation returns per-node `ShapeSummary` values, which the Angular Builder uses for port inspection and model diagnostics. Generated source and isolated workers continue to receive batch-inclusive runtime specifications derived from the canonical feature-only graph.

Shape inference traverses nodes in topological order and maps manifest shape-rule kinds to deterministic functions. It currently covers inputs, dense, flatten, reshape, convolution rank checks, pooling defaults and data formats, recurrent layers, embeddings, merge compatibility, concatenation axes, and Keras attention dimensions. Multi-port nodes use manifest port order, and graph validation rejects components that are not on an input-to-output path. Unknown or partly symbolic dimensions remain unknown. Failed inference is a blocking diagnostic, not a guessed result.

Effective outputs are resolved once from the active node catalog and graph topology. Input nodes, disabled or non-tensor ports, and unresolved manifests are excluded; the resulting node-port objects carry the output name, display label, port, and supported selectors. Compilation validation consumes those objects for per-output loss and metric mappings, activation/task diagnostics, and graph-derived output counts. Classification task metadata remains unknown when the output width is unknown.
