# Validation Rules

Last updated: 2026-09-01

Blocking errors include cycles, missing model inputs or outputs, missing required connections, dangling edges, unknown ports or definitions, duplicate node/edge/output identities, conflicting outputs, invalid parameters, explicit nulls for non-nullable parameters, undeclared parameters, disabled ports, declared port-rank and tensor-list element-kind contradictions, dtype or shape merge incompatibility, concatenation dimension mismatch, residual-shape mismatch, attention-dimension mismatch, unsafe mathematical expressions, and unsupported or unverified framework features. Parameter diagnostics identify the node and parameter whenever the source is known.

Warnings identify suspicious but potentially compilable graphs, including incomplete shape information. Compatibility notices describe target-specific limitations and are blocking for the selected execution profile. Informational diagnostics carry model summaries such as inferred tensor shapes, parameter counts, output mode, and non-binding model-type inference.

Shape inference covers the catalog's input, core, convolutional, recurrent, transformer, merge, and mathematical rules. It propagates symbolic or dynamic dimensions as unknown rather than fabricating concrete values; matrix operations use the batch-first rank contract enforced by the runtime emitters.
