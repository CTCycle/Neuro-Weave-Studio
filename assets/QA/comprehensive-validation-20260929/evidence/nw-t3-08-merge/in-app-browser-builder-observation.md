# NW-T3-08 in-app Browser observation

- Date: 2026-09-29
- URL: `http://127.0.0.1:6425/builder`
- Surface: Codex in-app Browser, visible Builder render
- Catalog response observed: 149 available nodes

The rendered Builder category navigation exposed `Merge 14 nodes`. Opening
that category displayed the live merge cards, including `merge.add`,
`merge.average`, `merge.concatenate`, `merge.divide`, `merge.dot`,
`merge.matrix_multiply`, `merge.maximum`, `merge.minimum`, `merge.multiply`,
`merge.residual`, `merge.skip`, `merge.split`, `merge.stack`, and
`merge.subtract`. The arithmetic cases used the rendered `Add`, `Average`,
`Divide`, `Maximum`, `Minimum`, `Multiply`, and `Subtract` entries through the
real Builder surface; no placeholder catalog was used.

The visible category title and node cards were inspected after the official
launcher reached frontend/backend readiness. This observation is visual
catalog evidence; runtime, persistence, and negative-shape results are stored
in the adjacent JSON receipts and screenshots.
