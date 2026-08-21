# Domain Ontology

Last updated: 2026-08-21

A Project owns one current Model Graph and a history of revisions. A Model Graph contains Node Instances, Edge Instances, model-level configuration, and export settings. A Node Instance references exactly one versioned Node Definition. A Model Execution Profile persists `execution_backend` and `keras_enabled`; its resolved implementation selects Keras or one backend-specific native adapter. An Exporter translates the canonical graph or compiled IR to that implementation. Validation Diagnostics describe errors, warnings, compatibility notices, or information.
