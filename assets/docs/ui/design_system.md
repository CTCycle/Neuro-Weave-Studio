# Design System

Last updated: 2026-09-01

NeuroWeave uses a dark technical drafting-board identity with slate surfaces, copper primary actions, cyan tensor ports, compact monospaced metadata, and restrained radii. It is intentionally distinct from ParaGraph. Design tokens live in `app/client/src/styles.css`. Color is never the only validation cue; labels, icons, counts, and text remain present.

Scrollbars share the same surface language across the page and nested overflow regions: thin geometry, a dark slate track, a muted slate thumb at rest, and a cyan thumb on hover or focus. Layout-critical scroll regions may reserve a stable gutter, but page-specific scrollbar colors should not override the shared treatment.
