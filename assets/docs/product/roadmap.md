# Roadmap

Last updated: 2026-09-03

Phase 1 is the implemented architecture editor. Phase 2 adds dataset metadata, training configuration, local training jobs, evaluation, and experiment tracking.

Phase 3 covers extensibility. The first safe custom-node authoring layer is now implemented as installation-wide templates that materialize into the existing canonical graph: mathematical templates map to the built-in safe-expression node, and sequential blocks expand into validated built-in manifests and connections. Public Hugging Face references can be verified and commit-pinned but remain non-executable metadata.

The remaining Phase 3 work is the trusted executable plugin model for artifact classes that cannot be represented by existing built-in manifests. That work must define signed package identity, deterministic ports and shape behavior, framework adapter capability, dependency isolation, security permissions, compatibility/version rules, and migration behavior. Repository code and arbitrary user Python/JavaScript must never bypass the canonical graph and validation layers.

Phase 4 adds deployment and additional exporters such as ONNX, JAX or Flax, TensorFlow Lite, TorchScript, Core ML, and OpenVINO. Every phase must extend or materialize into the canonical graph rather than introduce a parallel executable representation.
