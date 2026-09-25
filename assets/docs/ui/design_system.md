# Design System

Last updated: 2026-09-25

NeuroWeave uses a dark technical drafting-board identity with slate surfaces, accent primary actions, cyan tensor ports, compact monospaced metadata, and restrained radii. It is intentionally distinct from ParaGraph. Design tokens live in `app/client/src/styles.css`; component-local styles use the canonical `--bg`, `--surface`, `--surface-raised`, `--border`, `--text`, `--muted`, `--accent`, `--accent-bright`, and `--cyan` tokens plus the repository font stack (`'IBM Plex Mono', Consolas, monospace` for metadata). Color is never the only validation cue; labels, icons, counts, and text remain present.

Shared spacing, typography, control-height, and radius tokens extend that palette in the global stylesheet. The application header and page content use the same inset at a given viewport: 16px on wider layouts and 8px at 1024px and below. Page titles, descriptions, form controls, buttons, and icon actions share these tokens so route changes preserve the content origin and control proportions.

Scrollbars share the same surface language across the page and nested overflow regions: thin geometry, a dark slate track, a muted slate thumb at rest, and a cyan thumb on hover or focus. Layout-critical scroll regions may reserve a stable gutter, but page-specific scrollbar colors should not override the shared treatment.

Builder choice controls use the same slate surfaces, thin borders, cyan focus ring, accent selected state, and monospaced metadata. The Model Builder catalog reveals capability-driven filters progressively: category first, then current/backend compatibility, port type, family, and operation type. Unsupported capabilities remain discoverable and disabled or annotated. The Backend control uses an accessible themed listbox so its closed trigger, popup, hover state, selection check, and keyboard focus remain dark and consistent across browsers. Native Inspector selects use the dark color scheme and the same token palette.

Settings and dataset workspaces use the same drafting-board tokens with restrained borders. Settings uses a narrow left navigation rail and a wide flat content column; sections are separated by spacing and dividers instead of nested cards. Dataset workspace tabs use compact route navigation, while editor rows align a label, concise description, and control. Status remains explicit text with icons and counts, and responsive layouts collapse the Settings rail and dataset editor columns at smaller widths.

Builder Inspector hierarchy is intentionally dense: node identity and status are presented in a border-separated header, parameter groups use hairline section rhythm, and port specifications use compact direction badges and status text instead of nested cards. Cyan distinguishes inputs and copper distinguishes outputs, while required/missing state is also communicated through explicit labels and error text.

Input and output counts are graph-derived and are shown consistently on model cards and the selected-model workspace. The Inspector's batch-dimension preview is runtime-only and presentation-only. Compilation readiness, effective outputs, and blocking validation findings are shown as explicit text; a blocked compilation save does not update the saved revision. Rendered KaTeX equations fit the available width without creating a horizontal scrollbar or nested scroll region.
