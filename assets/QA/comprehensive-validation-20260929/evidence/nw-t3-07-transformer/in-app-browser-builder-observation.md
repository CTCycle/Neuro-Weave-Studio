# NW-T3-07 in-app Browser observation

Date: 2026-09-29
Route: `http://127.0.0.1:6425/builder`

The Codex in-app Browser rendered the live Builder before the repeatable serial
run. The Node catalog reported 149 items, the expanded Transformer category
reported 10 nodes, and the seven remaining Transformer definitions were
visible with their stable definition IDs:

- `transformer.attention_mask`
- `transformer.decoder_block`
- `transformer.encoder_block`
- `transformer.feed_forward`
- `transformer.positional_encoding`
- `transformer.residual`
- `transformer.stack`

Selecting Transformer Encoder Block opened the Properties inspector with the
expected attention, structure, shape, and behavior controls. This was a
rendered catalog/Inspector observation only; the repeatable save, reload,
validation, and runtime evidence is in the adjacent JSON receipts and PNG
captures.
