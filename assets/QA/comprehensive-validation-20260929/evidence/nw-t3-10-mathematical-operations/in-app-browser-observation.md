# NW-T3-10 in-app Browser observation

Date: 2026-09-29

The Codex in-app Browser was used first against the official frontend at
`http://127.0.0.1:6425/`. The live `/builder` accessibility/rendered state
showed the Node catalog with 149 available nodes, the Mathematical category
with 73 nodes, the graph canvas with input/output ports, and the Properties
Inspector. The mathematical catalog was opened and the selected operation
entries were visible before the repetitive network-backed matrix was run.

The 28-case serial Playwright run then supplied the repeatable API,
persistence, exact-profile runtime, negative-path, screenshot, and browser
diagnostic evidence. The Browser-first observation is intentionally kept
separate from the raw Playwright receipts.
