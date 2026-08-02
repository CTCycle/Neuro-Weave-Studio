# Node Manifest Specification

Last updated: 2026-07-30

Each manifest defines stable `id`, integer `version`, display name, category, description, input and output ports, parameter definitions, a shape-rule descriptor, framework mappings, UI metadata, deprecation state, and documentation reference. IDs use lower-case dotted names. Port and parameter IDs are unique inside the manifest. Frontend controls are derived from parameter types and constraints rather than node-specific forms.
