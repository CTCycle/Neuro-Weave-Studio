# NeuroWeave Documentation Index

Last updated: 2026-10-20

## Purpose

This is the repository's single documentation navigation root. Read it first, then follow the links for the narrowest relevant topic.

## Documentation model

- Architecture and ontology documents define product concepts and behavior contracts. Update the owning concept when implementation changes.
- The [project status ledger](project_status_ledger.md) is a compact snapshot of scoped evidence, current limitations, and open validation work. It is not a development diary or a second product ontology.
- The [validation strategy](development/validation_strategy.md) defines evidence rules. The [testing strategy](development/testing_strategy.md) describes executable checks and their limits.
- Generated logs, screenshots, manifests, and other run output belong under ignored assets/QA. Keep durable QA knowledge in the existing concept documents and status ledger rather than checking in large evidence trees.

## Product

- product/vision_and_scope.md: product boundary and MVP definition.
- product/user_personas.md: intended users and capability assumptions.
- product/user_flows.md: catalog, Builder, workspace, and export journeys.
- product/roadmap.md: staged expansion beyond architecture design.

## Architecture

- architecture/system_overview.md
- architecture/backend_architecture.md
- architecture/frontend_architecture.md: Angular application composition, graph ownership, and UI boundaries.
- architecture/dataset_management.md: external dataset registration, inspection, validation, catalog, and workspace boundaries.
- architecture/metadata_contract.md: runtime metadata states, on-demand discovery, and cache identity.
- architecture/graph_compiler.md
- architecture/model_execution_profile.md
- architecture/shape_inference.md
- architecture/persistence.md
- architecture/export_pipeline.md

## Ontology

- ontology/domain_ontology.md
- ontology/graph_entities.md
- ontology/tensor_and_shape_system.md
- ontology/node_taxonomy.md
- ontology/ports_and_connections.md
- ontology/framework_compatibility.md
- ontology/validation_rules.md
- ontology/serialization_and_versioning.md
- ontology/plugin_and_custom_node_model.md

## Nodes

- nodes/manifest_specification.md
- nodes/core_layers.md
- nodes/convolutional_layers.md
- nodes/recurrent_layers.md
- nodes/transformer_layers.md
- nodes/mathematical_operations.md
- nodes/merge_operations.md
- nodes/training_configuration.md

## Development

- development/coding_standards_python.md
- development/coding_standards_typescript.md
- development/testing_strategy.md
- development/validation_strategy.md
- development/contribution_guide.md
- development/adding_a_builtin_node.md

## UI

- ui/design_system.md
- ui/editor_interactions.md
- ui/model_workspaces.md
- ui/accessibility.md
- ui/error_and_validation_patterns.md

## Runtime

- runtime/installation.md
- runtime/startup.md
- runtime/configuration.md
- runtime/troubleshooting.md

## Decisions

- decisions/adr_001_frontend_framework.md
- decisions/adr_002_graph_library.md
- decisions/adr_003_canonical_model_representation.md
- decisions/adr_004_export_strategy.md

## Documentation rules

1. Keep this file as the only navigation root.
2. Use lower-case file names and keep each leaf focused on one concern.
3. Include a Last updated date in maintained documentation.
4. Keep the status ledger current without copying detailed QA history into it.
