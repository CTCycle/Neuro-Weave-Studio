# Editor Interactions

Last updated: 2026-08-21

Users drag catalog items onto the canvas, connect typed handles, select nodes or edges, edit node properties, and delete the current selection. The canvas supports pan, zoom, fit view, a minimap, snap-to-grid placement, and automatic layered layout. Connection attempts that would create a cycle, violate port kinds, or overfill a single-input port are rejected before the edge is created.

Copy and paste duplicate the selected node. Undo and redo cover node creation, connection creation, deletion, property changes, model configuration changes, node movement checkpoints, paste operations, and automatic layout. Keyboard commands use Ctrl or Command with Z, Shift+Z, Y, C, V, and L. Reduced-motion preferences disable nonessential animation.

The Builder keeps open models in a tab bar. Each tab owns a graph snapshot, metadata snapshot, saved baseline, and dirty state; switching tabs restores the selected session without losing another model's draft. Closing a dirty tab requires confirmation. New tabs create isolated drafts and never duplicate a persisted project ID.

The toolbar edits model name and description, selects the persisted execution backend, toggles Keras-enabled versus native implementation, and exposes Validate, Save & compile, Save model, undo, redo, and workspace actions. Keras is shown as an implementation mode, not as a competing backend.

The fixed editor workspace is a three-column layout: Node catalog, center workspace, and Inspector. The center contains the compilation summary, canvas, and bottom Validation panel. The validation splitter changes the panel height continuously while preserving a minimum canvas height; diagnostics scroll inside the panel instead of growing the document.
