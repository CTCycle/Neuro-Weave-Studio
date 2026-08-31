# Editor Interactions

Last updated: 2026-08-31

Users drag catalog items onto the Angular canvas, connect typed ports, select nodes or edges, edit node properties, and delete the current selection. Foblex supplies rendering and pointer gestures through the adapter; the neutral graph store remains authoritative. The canvas supports pan, zoom, fit view, a minimap, snap-to-grid placement, and automatic layered layout. Connection attempts that would create a cycle, violate port kinds, or overfill a single-input port are rejected before the edge is created.

Copy and paste duplicate the selected node. Undo and redo cover node creation, connection creation, deletion, property changes, model configuration changes, node movement checkpoints, paste operations, and automatic layout. Keyboard commands use Ctrl or Command with Z, Shift+Z, Y, C, V, and L. Reduced-motion preferences disable nonessential animation.

The Builder keeps open models in a tab bar. Each tab owns a graph snapshot, metadata snapshot, saved baseline, and dirty state; switching tabs restores the selected session without losing another model's draft. Closing a dirty tab requires confirmation. New tabs create isolated drafts and never duplicate a persisted project ID.

The toolbar edits the model name, opens a compact canonical description popover, selects the persisted native execution backend, explicitly toggles the optional Keras API layer, and exposes Validate, Save model, undo, redo, and workspace actions. Compilation configuration, validation, and source generation continue in the dedicated Compilation workspace. The backend selector remains the native execution choice; Keras is an optional abstraction over it, not a competing backend.

The fixed editor workspace is a three-column layout: Node catalog, center workspace, and Inspector. The center contains the model tab bar, canvas, and bottom Validation panel. The minimap can be hidden locally so it releases its canvas area. Focus mode keeps the same Angular/Foblex graph instance mounted, hides nonessential surrounding panels, and restores the normal layout with Escape or Exit full canvas without fitting or resetting the viewport. The validation splitter changes the panel height continuously while preserving a minimum canvas height; diagnostics scroll inside the panel instead of growing the document.
