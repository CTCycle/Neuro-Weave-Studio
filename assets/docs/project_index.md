# NeuroWeave Documentation Index

Last updated: 2026-08-02

## Purpose

This is the root navigation document for the repository. Read it first, then open only the narrowest leaf documents required for the current task.

## Product

- `product/vision_and_scope.md`: product boundary and MVP definition.
- `product/user_personas.md`: intended users and capability assumptions.
- `product/user_flows.md`: primary editor and export journeys.
- `product/roadmap.md`: staged expansion beyond architecture design.

## Architecture

- `architecture/system_overview.md`
- `architecture/backend_architecture.md`
- `architecture/frontend_architecture.md`
- `architecture/graph_compiler.md`
- `architecture/shape_inference.md`
- `architecture/persistence.md`
- `architecture/export_pipeline.md`

## Ontology

- `ontology/domain_ontology.md`
- `ontology/graph_entities.md`
- `ontology/tensor_and_shape_system.md`
- `ontology/node_taxonomy.md`
- `ontology/ports_and_connections.md`
- `ontology/framework_compatibility.md`
- `ontology/validation_rules.md`
- `ontology/serialization_and_versioning.md`
- `ontology/plugin_and_custom_node_model.md`

## Nodes

- `nodes/manifest_specification.md`
- `nodes/core_layers.md`
- `nodes/convolutional_layers.md`
- `nodes/recurrent_layers.md`
- `nodes/transformer_layers.md`
- `nodes/mathematical_operations.md`
- `nodes/merge_operations.md`
- `nodes/training_configuration.md`

## Development

- `development/coding_standards_python.md`
- `development/coding_standards_typescript.md`
- `development/testing_strategy.md`
- `development/contribution_guide.md`
- `development/adding_a_builtin_node.md`

## UI

- `ui/design_system.md`
- `ui/editor_interactions.md`
- `ui/accessibility.md`
- `ui/error_and_validation_patterns.md`

## Runtime

- `runtime/installation.md`
- `runtime/startup.md`
- `runtime/configuration.md`
- `runtime/troubleshooting.md`

## Decisions

- `decisions/adr_001_frontend_framework.md`
- `decisions/adr_002_graph_library.md`
- `decisions/adr_003_canonical_model_representation.md`
- `decisions/adr_004_export_strategy.md`

## Documentation rules

1. Preserve this index as the single navigation root.
2. Use lower-case file names.
3. Keep each leaf focused on one concern.
4. Update architecture, ontology, and user-facing documentation with implementation changes.
5. Include an explicit `Last updated` date in every document.
