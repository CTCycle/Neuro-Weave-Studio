# Plugin and Custom Node Model

Last updated: 2026-09-03

NeuroWeave now provides an installation-wide custom node definition library alongside the immutable built-in manifest catalog. Built-in node manifests remain owned by the application and cannot be edited or deleted. User-authored definitions are persisted separately, may be edited or removed, and never overwrite or shadow built-in identities.

Custom definitions support three authoring strategies:

- **Mathematical operation**: stores a LaTeX equation plus an explicit set of input symbols. The frontend supports direct LaTeX entry and a guided equation builder with the same rendered preview surface used by custom metrics and losses.
- **Composite node**: stores an ordered sequential composition of built-in node definitions. Component parameters are normalized against the selected built-in manifests so the definition keeps the complete parameter contract and cannot reference unknown parameters or unknown node identities.
- **Public repository reference**: stores public Hugging Face repository coordinates, revision, artifact type, and an optional entrypoint/class hint. Repository references are metadata only and do not enable `trust_remote_code`, import remote packages, or execute repository code in the FastAPI process.

These user definitions are an authoring/library concept, not an executable plugin mechanism. They are intentionally kept separate from the runtime `NodeCatalog` until a trusted execution contract exists for custom mathematical expressions, composite expansion, and repository-backed layers. The Model Builder therefore continues to compile and validate only canonical built-in manifests.

A future executable plugin must still declare a signed package identity, compatible app and schema versions, deterministic shape behavior, framework adapter capabilities, security permissions, and migration rules. Arbitrary Python or JavaScript embedded in project JSON remains forbidden.
