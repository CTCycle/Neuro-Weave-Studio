# Plugin and Custom Node Model

Last updated: 2026-09-03

NeuroWeave provides an installation-wide custom authoring-template library alongside the immutable built-in manifest catalog. Built-in `NodeManifest` definitions remain application-owned and cannot be edited or deleted. User-authored templates are persisted separately, may be edited or removed, carry their own monotonically increasing revision number, and never overwrite or shadow built-in identities.

The custom library supports three authoring strategies:

- **Mathematical operation**: the user authors a LaTeX-style single-input tensor expression. The server normalizes it into NeuroWeave's existing bounded safe-expression language and rejects unsupported names, syntax, or operations before persistence. In the Model Builder the template materializes as the canonical built-in `math.expression` manifest with the validated expression parameter.
- **Sequential block**: the user selects an ordered sequence of active built-in operation manifests. Parameter values are normalized and validated against the same manifest types, options, nullability, conditional rules, safe-expression checks, and numeric/list constraints used by normal graph authoring. In the Model Builder the template expands into ordinary built-in graph nodes and canonical connections. The persisted model therefore contains only stable built-in manifest references, never a private composite runtime object.
- **Public repository reference**: the user supplies public Hugging Face model coordinates and a revision. Creation verifies that the public model/revision can be resolved and stores the exact resolved commit plus available library and pipeline metadata. This is reference metadata only. Repository code is not imported or executed, `trust_remote_code` is never enabled, and the reference is not insertable into the Builder until NeuroWeave has a trusted execution adapter contract for that artifact class.

This materialization boundary preserves ADR 003. The canonical model graph continues to contain only node instances backed by validated built-in manifests. Custom templates are authoring conveniences that produce canonical graph content before model persistence; they do not create a parallel executable graph schema. Consequently, editing or deleting a template does not retroactively change models that were already materialized from it.

Template revisions identify edits to the reusable authoring definition. They are not node-manifest versions and are never persisted as graph `definition_version` values. Built-in node IDs and versions remain the only executable definition identities in saved graphs.

A future executable plugin system may extend this model, but it must declare a signed package identity, compatible app and schema versions, deterministic shape behavior, framework adapter capabilities, security permissions, and migration rules. Arbitrary Python or JavaScript embedded in project JSON remains forbidden.
