# User Checklists

Last updated: 2026-08-02

## End-to-end editor checklist

1. Start the application and confirm the frontend loads.
2. Confirm `GET /health` returns a healthy response.
3. Create or open a project and choose its target framework.
4. Add nodes and connect only compatible ports.
5. Configure required parameters and model settings.
6. Run backend validation and resolve blocking diagnostics.
7. Review inferred shapes, parameter counts, and generated source.
8. Save, reopen, and confirm the current revision is restored.
9. Export JSON or source only after reviewing the result.

## Before sharing an artifact

- Confirm the project name and description contain no secrets.
- Confirm the selected framework is the intended target.
- Confirm validation has no blocking diagnostics.
- Confirm generated source matches the selected framework.
- Remember that `.keras` output is initialized and untrained.
- Treat database reset as destructive and keep backups of any project bundle you
  need to retain.

## Current non-goals

Training, datasets, experiment tracking, deployment, learned weights, ONNX or
TorchScript export, cross-framework conversion, and executable third-party
plugins are not part of the current MVP.
