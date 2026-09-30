# NW-T3-18 and NW-T3-19 Output Contracts and Graph Integrity

Date: 2026-09-30
Run: `NW-CV-20260930-T3-18-19-OUTPUT-GRAPH`
Source baseline: `416c158c61349d7a1106e71e64fba4174d3178a5` before the scoped test/evidence changes

## Result

`NW-T3-18` and `NW-T3-19` PASS for the exercised live contract classes. Evidence consists of 11 focused backend validation/profile tests, one focused output audit, one explicit-output browser workflow, one graph-integrity browser workflow, and the dedicated dangling-endpoint regression. The focused backend result is recorded in [logs-nw-t3-18-19-backend-focused.txt](logs-nw-t3-18-19-backend-focused.txt); the additional dangling-endpoint result is in [logs-nw-t3-19-dangling-backend.txt](logs-nw-t3-19-dangling-backend.txt).

## NW-T3-18 output/task contracts

The backend evidence covers inferred terminal outputs, multiple explicit output bindings, conditional recurrent state outputs, incompatible output/task combinations, and rejection of unreachable model inputs. The browser evidence covers the explicit-output enable/remove transitions and persists/rechecks the resulting graph state. The live UI run is in [logs-nw-t3-18-output-ui.txt](logs-nw-t3-18-output-ui.txt), with receipts under [evidence/nw-t3-18-19](evidence/nw-t3-18-19).

## NW-T3-19 graph and port integrity

The generated [edge campaign](evidence/nw-t3-19-generated-edge-campaign.json) freezes every live definition and port, then partitions 22,630 output/input candidates by manifest kind/rank/cardinality instead of attempting a literal pairwise execution. It records 17,014 statically compatible-rank candidates, 124 static rank mismatches, 5,492 incompatible-kind candidates, 5,115 fan-in targets, and 17,515 single-target ports. Dynamic cases are covered by focused tests and browser workflows:

- positive compatible edges, fan-out, and variadic fan-in;
- rank and dimension mismatch;
- dtype incompatibility;
- cardinality violation;
- cycle creation;
- dangling source and target endpoints (`dangling_connection`);
- deleted/unreachable graph components; and
- execution-profile edge restrictions.

The browser workflow also validates incompatible shape rejection, parameter repair, save, cycle rejection, typed incompatible-port rejection, node deletion, and revalidation. The final campaign remains bounded to the frozen catalog and the supported CPU profiles exercised by the existing harness; full worker, export, accessibility, and Tier 4 capability closure remain separate.
