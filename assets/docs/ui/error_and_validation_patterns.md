# Error and Validation Patterns

Last updated: 2026-08-27

Blocking errors use `error`, suspicious conditions use `warning`, implementation limits use `compatibility`, and summaries use `info`. Node diagnostics appear on the node and in the inspector. Global diagnostics appear below the canvas and can select the affected node. API failures produce dismissible notices. Unsupported source or binary export must return a clear 4xx or structured job failure, never an empty artifact.

Validation has two distinct layers. Static validation checks graph topology, typed ports, parameters, shape propagation, the selected execution profile and canonical Keras/native configuration branch, node implementation support, and generated-source eligibility. Runtime validation is isolated per selected Keras or native worker: it reports dependency, device, compiler, dtype, tensor-shape, process, and artifact diagnostics. An unavailable native compiler is an explicit capability diagnostic; the runtime never silently switches to Keras.

The Builder combines backend diagnostics with local graph diagnostics in one resizable bottom panel. The panel is collapsed until a result or supplemental diagnostic exists, opens automatically for blocking findings, deduplicates repeated messages, and lets the user select the affected node or edge.
