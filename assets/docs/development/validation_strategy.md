# Validation Strategy

Last updated: 2026-10-21

## Purpose

This document defines how to make bounded validation claims. The [status ledger](../project_status_ledger.md) holds the current evidence snapshot; architecture and ontology documents own the product contracts.

## Evidence rules

- Identify the application source revision, environment, exact execution profile, and scenarios for every result.
- A result applies only to the named scenario and profile. A static mapping, metadata response, or generated source does not prove browser rendering or live runtime execution.
- Treat the backend and API as part of profile identity. Do not accept a fallback backend or API as proof for the selected profile.
- Separate product failures from missing dependencies, toolchain limits, and test-harness failures. Report injected failures as injected; do not describe them as natural runtime crashes.
- Keep browser evidence distinct from API evidence, and retain the cleanup result when a check creates projects, jobs, processes, or temporary data.
- Do not include credentials or user data in logs or retained evidence.

## Validation workflow

1. Read the owning concept document and identify its stated contract and known limitations.
2. Select focused positive, negative, and recovery cases that exercise the changed boundary.
3. Use isolated, disposable state and record the exact profile and source revision.
4. Run the relevant automated or browser check and inspect its actual output.
5. Classify failures before changing status; rerun the failing case and a nearby valid or invalid case after a justified fix.
6. Update the behavior contract only when behavior changes, then update the status ledger with the narrow evidence and remaining boundary.

## Route evidence to its concept owner

| Validation area | Contract owner |
| --- | --- |
| Graph entities, ports, shape, and validation | [Graph entities](../ontology/graph_entities.md), [ports and connections](../ontology/ports_and_connections.md), [tensor and shape system](../ontology/tensor_and_shape_system.md), [validation rules](../ontology/validation_rules.md) |
| Node mappings and execution profiles | [Framework compatibility](../ontology/framework_compatibility.md), [model execution profile](../architecture/model_execution_profile.md) |
| Source, compilation, binary export, and recovery | [Graph compiler](../architecture/graph_compiler.md), [export pipeline](../architecture/export_pipeline.md) |
| Project and dataset state | [Persistence](../architecture/persistence.md), [serialization and versioning](../ontology/serialization_and_versioning.md), [dataset management](../architecture/dataset_management.md) |
| Browser interaction and accessibility | [Editor interactions](../ui/editor_interactions.md), [accessibility](../ui/accessibility.md) |

## Result vocabulary

Product components and areas use VALIDATED only for the named passed scope. Use PARTIAL when a meaningful subset passes and important scenarios remain. Use UNVALIDATED when implementation exists without enough evidence. Use NOT_IMPLEMENTED when the product capability is absent. Bounded campaign gates use the gate vocabulary PASS, PARTIAL, FAIL, UNTESTED, and BLOCKED, defined in the [status ledger](../project_status_ledger.md). Record the next useful check in the ledger; do not maintain a separate gate history there.
