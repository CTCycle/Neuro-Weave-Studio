# Accessibility

Last updated: 2026-11-11

Inputs have labels, dialogs expose roles, focus remains visible through native controls and links, and reduced-motion preferences disable animation. Diagnostics use text severity and not only color. The application remains usable in smaller windows: Catalog and Inspector become drawers at 1024px and below, toolbar actions reflow, and page content scrolls when it exceeds the available space. Anchored menus remain within the viewport, and non-canvas targets are at least 24×24px. The graph canvas is a legitimate two-dimensional workspace exception under WCAG Reflow.

The graph also exposes an alternate keyboard interaction model. Each node is a focusable `group`; arrow keys move focus to the nearest node in that direction and Enter or Space selects it. Input and output ports are focusable `button` targets. Enter on an output selects a connection source, Enter on an input reuses the same server-side validation path as pointer dragging, and Escape cancels the pending connection. The live instruction is exposed through a polite status region, and source/compatible-target states have non-color focus styling.

The Nodes route uses a contained two-pane workspace below global navigation: the catalog toolbar remains visible while the catalog list scrolls independently, and the selected-node details body has its own scroll region. At narrow widths the panes and structured detail fields collapse without horizontal overflow. Context-menu actions expose native disabled state and explanatory titles when an operation is unavailable.

`app/client/e2e/accessibility.spec.ts` runs a browser baseline across the core routes: document language/title, main/navigation landmarks, visible interactive accessible names, and visible focus styling. This is regression coverage, not a claim of full WCAG conformance. A complete WCAG/assistive-technology certification still requires manual review with a screen reader and the supported browser/OS matrix, including the graph's two-dimensional interaction exception.
