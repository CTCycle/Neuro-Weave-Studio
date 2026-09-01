# Design System

Last updated: 2026-09-01

NeuroWeave uses a dark technical drafting-board identity with slate surfaces, copper primary actions, cyan tensor ports, compact monospaced metadata, and restrained radii. It is intentionally distinct from ParaGraph. Design tokens live in `app/client/src/styles.css`. Color is never the only validation cue; labels, icons, counts, and text remain present.

Scrollbars share the same surface language across the page and nested overflow regions: thin geometry, a dark slate track, a muted slate thumb at rest, and a cyan thumb on hover or focus. Layout-critical scroll regions may reserve a stable gutter, but page-specific scrollbar colors should not override the shared treatment.

Builder choice controls use the same slate surfaces, thin borders, cyan focus ring, copper selected state, and monospaced metadata. The Backend control uses an accessible themed listbox so its closed trigger, popup, hover state, selection check, and keyboard focus remain dark and consistent across browsers. Native Inspector selects use the dark color scheme and the same token palette.

Builder Inspector hierarchy is intentionally dense: node identity and status are presented in a border-separated header, parameter groups use hairline section rhythm, and port specifications use compact direction badges and status text instead of nested cards. Cyan distinguishes inputs and copper distinguishes outputs, while required/missing state is also communicated through explicit labels and error text.
