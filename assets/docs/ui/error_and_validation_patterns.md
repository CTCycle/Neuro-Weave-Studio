# Error and Validation Patterns

Last updated: 2026-09-17

Blocking errors use `error`, suspicious conditions use `warning`, implementation limits use `compatibility`, and summaries use `info`. Node diagnostics appear on the node and in the inspector. Global diagnostics appear below the canvas and can select the affected node. API failures produce dismissible notices. Compatibility details use stable reason codes such as `missing_implementation`, `unsupported_implementation`, `unverified_implementation`, `backend_excluded`, `unsupported_parameter`, and `unsupported_parameter_option`. Unsupported source or binary export must return a clear 4xx or structured job failure, never an empty artifact.

Validation has two distinct layers. Static validation checks graph topology, typed ports, parameters, shape propagation, the selected execution profile and canonical Keras/native configuration branch, node implementation support, and generated-source eligibility. Runtime validation is isolated per selected Keras or native worker: it reports dependency, device, compiler, dtype, tensor-shape, process, and artifact diagnostics. An unavailable native compiler is an explicit capability diagnostic; the runtime never silently switches to Keras.

The Builder combines backend diagnostics with local graph diagnostics in one resizable bottom panel. The panel is collapsed until a result or supplemental diagnostic exists, opens automatically for blocking findings, deduplicates repeated messages, and lets the user select the affected node or edge.

Tensor inspection failures are explicit and recoverable. Stable codes include `model_not_valid`, `revision_conflict`, `runtime_unavailable`, `dynamic_dimension_unresolved`, `unsupported_dtype`, `node_uninspectable`, `capture_unsupported`, `session_expired`, `slice_out_of_bounds`, `payload_limit`, and `execution_failed`. The modal shows the explanation instead of silently opening an empty view; expiry and cleanup failures can be retried or dismissed. Sampled values/statistics are labeled as sampled, and the untrained/random-initialization warning remains visible for the entire session.
