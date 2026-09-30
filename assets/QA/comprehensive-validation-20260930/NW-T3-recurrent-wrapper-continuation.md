# NW-T3 Recurrent Wrapper Continuation

Date: 2026-09-30
Run: `NW-CV-20260930-T3-RECURRENT-WRAPPER`
Source baseline: `416c158c61349d7a1106e71e64fba4174d3178a5` before the scoped test/evidence changes

## Result

The corrected wrapper acceptance run passed **6/6** serial Builder/browser cases in 1.6 minutes. Evidence is in [logs-nw-t3-recurrent-wrapper-playwright.txt](logs-nw-t3-recurrent-wrapper-playwright.txt) and [evidence/nw-t3-recurrent-wrapper](evidence/nw-t3-recurrent-wrapper).

The prior `64`/`68` expectation was a validation-spec defect. For input width 3 and four units, one standard LSTM has `4 × (3×4 + 4×4 + 4) = 128` parameters; bidirectional LSTM therefore has `256`, and the two-layer stack has `272`. Static inference and the Keras/PyTorch CPU runtime agree at `256`/`272`. All merge modes, invalid stack-layer repair, persistence, and runtime assertions passed with those corrected criteria.

No implementation change was made to force the obsolete counts. The old PARTIAL report remains historical; this continuation is the active acceptance evidence and removes the stale expectation from the campaign ledger.
