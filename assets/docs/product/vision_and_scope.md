# Vision and Scope

Last updated: 2026-10-18

NeuroWeave Studio lets users build machine-learning architectures as validated directed acyclic graphs. The current product includes the Model Builder, versioned project persistence, profile-aware source generation and artifact export, dataset registration and inspection, declarative processing/split settings, and safe template-based custom-node authoring.

Dataset registrations reference user-owned local sources and store bounded definitions and inspection metadata; they do not copy source bytes or materialize partitions. Public repository references are pinned metadata, not executable code.

Functional Validation and Tensor Inspection use synthetic inputs and initialized weights. Binary exports are untrained. Training, evaluation, prediction serving, dataset materialization, deployment, trusted executable plugins, arbitrary custom code, and general cross-framework conversion are outside the current product scope.

The canonical graph and its selected execution profile remain the shared representation for validation, source generation, and export. See [dataset management](../architecture/dataset_management.md), [export pipeline](../architecture/export_pipeline.md), and the [custom-node model](../ontology/plugin_and_custom_node_model.md) for the respective contracts.
