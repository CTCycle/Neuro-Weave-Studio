# Workflow Editor

Last updated: 2026-08-02

## Canvas basics

- Search or filter the node catalog, then drag a manifest onto the canvas.
- Connect an output handle to a compatible input handle.
- Pan, zoom, select, delete, copy, paste, undo, redo, and run deterministic
  automatic layout from the canvas controls.
- Select a node to edit its label and parameters in the properties inspector.
- Use the model configuration panel for optimizer, loss, metric, initializer,
  regularizer, constraint, mixed-precision, and seed values.

## Validation loop

Validation should be run before compilation or export. The panel groups errors
and warnings and displays inferred tensor shapes and parameter counts. Errors
must be fixed before source generation. Warnings remain visible so the user can
decide whether a disconnected or non-terminal node is intentional.

The client performs immediate cycle and connection checks for responsiveness;
the backend repeats the checks and adds parameter, shape, and framework
compatibility diagnostics.

## Compile and review

Select a framework in the toolbar, compile the graph, and review the generated
source in the code preview. Keras and TensorFlow-through-Keras share the Keras
adapter terminology; PyTorch is a separate source adapter with a narrower
supported subset. Unsupported nodes are reported as compatibility diagnostics,
not silently rewritten.
