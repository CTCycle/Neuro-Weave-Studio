# Roadmap

Last updated: 2026-10-18

## Phase 1: Architecture editor

Implemented: graph authoring and validation, versioned project persistence, profile-aware source generation, and capability-gated artifacts. The dataset metadata foundation includes external-source registration, bounded inspection and revalidation, persisted definitions, and Overview/Validation/I/O/Processing/Split workspaces. Processing currently configures source interpretation; split settings remain declarative. Safe mathematical and graph-template custom-node authoring is also implemented.

## Phase 2: Data execution and training

Add preprocessing, partition materialization, training jobs, optimizer steps, checkpoints, evaluation, and experiment tracking. These capabilities require explicit dataset and runtime contracts and are not implemented by the current draft Training workspace.

## Phase 3: Extensibility

Trusted executable plugins remain future work. Define signed package identity, deterministic ports and shape behavior, framework capabilities, dependency isolation, permissions, version compatibility, and migration before executing third-party code. Public Hugging Face references remain commit-pinned metadata and must never bypass the canonical graph or validation layers.

## Phase 4: Deployment and additional targets

Add deployment workflows and additional export targets such as TensorFlow Lite, Core ML, and OpenVINO. Existing ONNX and framework-native artifact formats belong to the current capability-gated export pipeline. Future exporters must preserve the canonical graph, selected profile, verification, and atomic publication contracts.
