# Error and Validation Patterns

Last updated: 2026-07-30

Blocking errors use `error`, suspicious conditions use `warning`, framework limits use `compatibility`, and summaries use `info`. Node diagnostics appear on the node and in the inspector. Global diagnostics appear below the canvas and can select the affected node. API failures produce dismissible notices. Unsupported export must return a clear 4xx or 501 response, never an empty artifact.
