# NW-T2-05 in-app Browser observation

Date: 2026-09-24

After the E2E workflows completed and the leftover disposable fixture was
removed from the isolated validation database, the Codex in-app Browser showed
the `/models` route with `0 / 0 models` and the empty-library message. The
observed screen confirms the preview loaded and no test project remained in the
workspace.

The inspector rendering itself is captured in
`nw-t2-05/nw-t2-05-builder-sequence-heatmap.png`; the live browser workflow
asserts the selected recurrent node, axes 1 and 2, and three rows by four
columns. Fixture cleanup is recorded in
`../logs/nw-t2-05-fixture-cleanup-20260924.log`.
