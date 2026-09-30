# NW-T3-15 through NW-T3-17 Mathematical Operations

Date: 2026-09-30
Run: `NW-CV-20260930-T3-15-17-MATHEMATICAL`
Source baseline: `416c158c61349d7a1106e71e64fba4174d3178a5` before the scoped test/evidence changes

## Result

`NW-T3-15`, `NW-T3-16`, and `NW-T3-17` PASS for the live definitions assigned by the frozen catalog crosswalk. The final serial browser aggregate passed **32/32** cases in 7.7 minutes on the official launcher, using the Keras API over PyTorch CPU and native PyTorch CPU profiles. Every case exercised Builder materialization, typed parameters, static shape inference, source/runtime execution, save/reopen, and the invalid-state repair path supplied by the shared slice harness.

The live catalog contained 149 definitions and 73 Mathematical definitions with normalized fingerprint `acf538fb04f50552e1577299aa3f0cc09511d4e7d0937f73aa8418125a3a9ef8`. Ownership is machine-readable in [catalog-crosswalk.json](evidence/catalog-crosswalk.json):

| Slice | Definitions | Families |
| --- | ---: | --- |
| NW-T3-15 | 6 | divide, dot, maximum, minimum, multiply, subtract |
| NW-T3-16 | 8 | concatenate, flatten, ravel, repeat, reshape, squeeze, stack, tile |
| NW-T3-17 | 2 | stop-gradient, take |

The crosswalk contains no missing or duplicate Mathematical definition IDs. The final aggregate is [logs-nw-t3-15-17-mathematical-final-aggregate.txt](logs-nw-t3-15-17-mathematical-final-aggregate.txt), with per-case JSON and rendered receipts under [evidence/nw-t3-15-17-mathematical](evidence/nw-t3-15-17-mathematical).

## Boundary evidence

- Negative axes were exercised for concatenate, repeat, reshape-adjacent operations, and take where the live contract supports them.
- Rank-changing cases covered flatten/ravel, squeeze, stack, and tile; repeat and tile counts were materialized through Builder parameters.
- Dot and matrix-shaped arithmetic covered inner-dimension compatibility and exact batch-inclusive runtime shapes.
- `math.stop_gradient` retained the expected shape and typed mapping; safe-expression rejection remains covered by the active NW-T3-10 acceptance case.
- Browser request/page-error receipts are empty for the final aggregate, and the slice cleanup hooks removed created projects and job state.

The acceptance boundary is intentionally exact-profile and CPU-scoped. Full six-profile Builder/runtime/export coverage, PyTorch Inductor, worker concurrency, and accessibility remain outside this Tier 3 campaign.
