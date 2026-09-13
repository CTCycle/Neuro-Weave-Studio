# Validation Rules

Last updated: 2026-09-13

Blocking errors include cycles, missing model inputs or outputs, missing required connections, dangling edges, unknown ports or definitions, duplicate node/edge/output identities, conflicting outputs, invalid parameters, explicit nulls for non-nullable parameters, undeclared parameters, disabled ports, declared port-rank and tensor-list element-kind contradictions, dtype or shape merge incompatibility, concatenation dimension mismatch, residual-shape mismatch, attention-dimension mismatch, impossible reshape targets, unsafe mathematical expressions, and unsupported or unverified framework features. Shape failures identify the affected node, the shape-rule reason, and the parameter that caused it whenever the source is known.

Compatibility is resolved against exactly one of the six static profiles. `missing_implementation`, `unsupported_implementation`, `unverified_implementation`, and `backend_excluded` identify node-level mapping failures; parameter mappings use corresponding unsupported, unverified, fixed-value, disabled, and unsupported-option diagnostics. Existing incompatible nodes, edges, and stored values are retained for repair, but the complete graph and selected profile must validate before source generation, compilation, export, or execution. Validation results are discarded when their graph revision or profile no longer matches.

Warnings identify suspicious but potentially compilable graphs, including incomplete shape information. Compatibility notices describe target-specific limitations and are blocking for the selected execution profile. Informational diagnostics carry model summaries such as inferred tensor shapes, parameter counts, output mode, and non-binding model-type inference.

Shape inference covers the catalog's input, core, convolutional, recurrent, transformer, merge, and mathematical rules. It propagates symbolic or dynamic dimensions as unknown rather than fabricating concrete values; matrix operations use the batch-first rank contract enforced by the runtime emitters.
