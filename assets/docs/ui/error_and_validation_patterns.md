# Error and Validation Patterns

Last updated: 2026-08-21

Blocking errors use `error`, suspicious conditions use `warning`, implementation limits use `compatibility`, and summaries use `info`. Node diagnostics appear on the node and in the inspector. Global diagnostics appear below the canvas and can select the affected node. API failures produce dismissible notices. Unsupported source or binary export must return a clear 4xx or structured job failure, never an empty artifact.

Validation has two distinct layers. Static validation checks graph topology, typed ports, parameters, shape propagation, the selected execution profile, node implementation support, and generated-source eligibility. Runtime validation is the isolated Keras smoke/export worker path: it reports dependency, device, dtype, tensor-shape, process, and artifact diagnostics. Native-only profiles retain static and source validation where implemented; native runtime inference is shown as unavailable with `not_implemented` rather than silently using Keras.

The Builder combines backend diagnostics with local graph diagnostics in one resizable bottom panel. The panel is collapsed until a result or supplemental diagnostic exists, opens automatically for blocking findings, deduplicates repeated messages, and lets the user select the affected node or edge.
