# NW-T3-06 in-app Browser observation

- Date: 2026-09-29
- Branch: `develop`
- Git HEAD: `42666a8374599e27367fcc2c9ae2a1ac77798ee7`
- URL: `http://127.0.0.1:6432/builder`
- Surface: Codex in-app Browser, real Builder route

The rendered Builder opened with the PyTorch backend selected and the Keras toggle enabled. The Catalog panel rendered `149 available` nodes. Expanding the Transformer category rendered `10 nodes`, including these three attention entries as real catalog buttons:

- `Self-attention` — `attention.self`
- `Cross-attention` — `attention.cross`
- `Multi-head Attention` — `attention.multi_head`

The Browser accessibility tree also exposed the Builder controls, graph canvas, Catalog, Inspector, Save model, and Graph Health sections. This observation is a UI/catalog check only; graph materialization, connection validation, persistence, and runtime execution are covered by the subsequent serial Playwright slice.
