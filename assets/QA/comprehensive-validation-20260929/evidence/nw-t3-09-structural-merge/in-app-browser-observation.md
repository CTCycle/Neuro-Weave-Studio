# NW-T3-09 in-app Browser observation

Date: 2026-09-29

Using the Codex in-app Browser against the official preview at
`http://127.0.0.1:6425/builder`, the live Model Builder rendered the Node
catalog with 149 available nodes. Opening the Merge category rendered all 14
live entries: Add, Average, Concatenate, Divide, Dot Product, Matrix
Multiplication, Maximum, Minimum, Multiply, Residual Connection, Skip
Connection, Split, Stack, and Subtract.

Activating Split materialized its Inspector controls. Changing Sections from
2 to 3 updated the live accessibility tree and Inspector connection list to
Output 0, Output 1, and Output 2. The canvas accessibility labels use the
direction suffix `(output)` while the persisted graph identities are
`output_0`, `output_1`, and `output_2`; the authoritative Playwright slice
asserts both forms.

The Browser tab was closed after observation. The durable rendered catalog and
per-case Builder/runtime screenshots are in this directory.
