# Node Manifest Specification

Last updated: 2026-07-30

Each manifest defines stable `id`, integer `version`, display name, category, description, input and output ports, parameter definitions, a shape-rule descriptor, v2 code-target mappings, UI metadata, deprecation state, and documentation reference. IDs use lower-case dotted names. Port and parameter IDs are unique inside the manifest. Every one of the 85 manifests is exposed through a complete safe parameter contract: each parameter has a description, group, nullable/list metadata where applicable, constraints, and serializable options. Frontend controls are derived from those definitions and render every parameter (the node label owns the layer `name`). Arbitrary callables, custom Python objects, and arbitrary code remain excluded.
