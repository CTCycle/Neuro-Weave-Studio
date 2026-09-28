# In-app Browser observation: recurrent wrapper run

Date: 2026-09-28
URL observed: `http://127.0.0.1:6429/`

The Codex in-app Browser was opened against the official launcher preview before
the authoritative Playwright run. The Models shell rendered successfully and
showed the live empty-project state. The browser tab was used for visual
observation only; the Playwright run remains authoritative for catalog
selection, Builder rendering, diagnostics, persistence, runtime results, and
browser-error collection. The tab was temporary and was not treated as a
durable deliverable.
