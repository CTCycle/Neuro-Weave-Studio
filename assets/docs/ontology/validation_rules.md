# Validation Rules

Last updated: 2026-10-18

Static validation applies to one canonical graph and one selected execution profile. It checks topology, identity, ports, parameters, compatibility, outputs, and shape contracts before source generation, compilation, export, or runtime work.

Blocking errors include cycles, missing model inputs or outputs, missing required connections, dangling edges, unknown ports or definitions, duplicate node/edge/output identities, conflicting outputs, invalid or undeclared parameters, explicit nulls for non-nullable parameters, disabled ports, declared port-rank or tensor-list element-kind contradictions, dtype or shape merge incompatibility, concatenation dimension mismatch, residual-shape mismatch, attention-dimension mismatch, impossible reshape targets, unsafe mathematical expressions, and unsupported or unverified framework features. When known, shape diagnostics identify the affected node, the reason, and the responsible parameter.

Compatibility resolves against exactly one of six static profiles: Keras over TensorFlow, PyTorch, or JAX; or native TensorFlow, PyTorch, or JAX. Missing, unsupported, unverified, and backend-excluded node mappings have distinct reason codes. Parameter mappings use corresponding unsupported, unverified, fixed-value, disabled, and unsupported-option diagnostics. Existing incompatible nodes, edges, and values remain in the graph for repair, but the complete graph and selected profile must validate before code generation or execution proceeds. No profile silently falls back to another API or backend.

A validation result is current only while its graph revision and execution profile match the current model. Stale results are discarded. Warnings identify suspicious but potentially compilable graphs, including incomplete shape information. Compatibility notices explain target-specific limits; they block work when they invalidate the selected profile. Informational diagnostics carry model summaries such as inferred shapes, parameter counts, output mode, and non-binding model-type inference.

Shape inference propagates symbolic or dynamic dimensions as unknown rather than inventing concrete values. Matrix operations use the batch-first rank contract enforced by runtime emitters. See the [tensor and shape system](tensor_and_shape_system.md) for the shape model.

## Related contracts

- [Graph entities](graph_entities.md) and [ports and connections](ports_and_connections.md) define the graph structure checked here.
- [Framework compatibility](framework_compatibility.md) defines the manifest mappings and profile resolution.
- [Model execution profile](../architecture/model_execution_profile.md) owns runtime and compiler selection.
- [Export pipeline](../architecture/export_pipeline.md) owns artifact capability and publication checks.
