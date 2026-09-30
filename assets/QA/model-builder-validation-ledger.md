# Model Builder validation ledger

Last updated: 2026-09-30

Preparation status: Tier 3 campaign closure recorded for the frozen 2026-09-30 catalog revision. NW-T3-01 through NW-T3-19 and the separately tracked recurrent-wrapper continuation are PASS within the exact evidence boundaries stated below. Tier 4, Tier 5, full six-profile/export coverage, worker concurrency, and accessibility remain separate.

## Purpose

This is the working ledger for consolidating Model Builder confidence across:

- every built-in node definition;
- every declared parameter and backend-specific parameter capability;
- every declared port/connection shape that can legally compose;
- representative multi-node graph families, branches, fan-in, fan-out, and multi-output graphs;
- all six execution profiles;
- Builder interaction, static validation, save/reopen, source generation, functional validation, and export boundaries.

The ledger is intentionally evidence-scoped. A node is not “green” merely because its manifest parses or a unit test covers its adapter. A profile is not “green” merely because capability discovery says that a mapping exists.

## Status vocabulary

Use these row-level values:

| Value | Meaning |
| --- | --- |
| PENDING | In scope and not yet run for the current run ID. |
| PASS | The required evidence for this row and profile is attached and matches the current revision. |
| EXPECTED_UNSUPPORTED | The manifest/profile contract says the path is unavailable and the Builder/API blocks it with the expected reason. |
| BLOCKED | The path should be tested, but the runtime, compiler, device, dependency, or environment is unavailable. Record the exact blocker. |
| FAIL | A path expected to work failed, or an expected diagnostic/fail-closed boundary was not observed. |
| STALE | Historical evidence exists, but it does not match the current revision or environment. Re-run before using it as closure evidence. |
| N/A | Structurally not applicable; explain why rather than leaving the cell blank. |

Use the project status ledger’s canonical component statuses when summarizing a completed run. Do not promote a component to VALIDATED from a PENDING, STALE, EXPECTED_UNSUPPORTED, or BLOCKED row.

## Source-of-truth and run setup

At the beginning of every run, freeze and record:

1. branch, commit, working-tree state, and launcher command;
2. the normalized response from /api/nodes/catalog, including count, category counts, definition IDs, versions, ports, parameters, and mappings;
3. a deterministic catalog fingerprint based on the sorted normalized catalog;
4. /api/metadata runtime and compiler discovery for the selected backend/API pair;
5. Python/framework/exporter versions, device, compiler/toolchain state, and test database identity;
6. the exact execution profile for each graph: execution_backend plus keras_enabled;
7. the graph/project ID, revision, validation response, job ID, generated-source identity, and artifact IDs where applicable.

The checked-in loader returned 149 manifests on 2026-09-21:

| Category | Seed count |
| --- | ---: |
| inputs | 8 |
| core | 14 |
| convolutional | 18 |
| recurrent | 5 |
| transformer | 10 |
| merge | 14 |
| mathematical | 73 |
| outputs | 7 |
| Total | 149 |

This is only the seed roster for this ledger. The manifest specification and framework-compatibility document contain different historical counts, so every future run must use the live normalized catalog and record the count/fingerprint instead of assuming that 149 remains current.

## Six profile matrix

The profile is part of the test identity. Never infer it from a request-local preference or accept a different worker/backend as a fallback.

| Profile key | execution_backend | keras_enabled | Required runtime lane |
| --- | --- | ---: | --- |
| K/TensorFlow | tensorflow | true | Keras API using TensorFlow |
| K/PyTorch | pytorch | true | Keras API using PyTorch |
| K/JAX | jax | true | Keras API using JAX |
| N/TensorFlow | tensorflow | false | Native TensorFlow |
| N/PyTorch | pytorch | false | Native PyTorch |
| N/JAX | jax | false | Native JAX/Flax |

For every profile, record these gates independently:

| Gate | Required evidence |
| --- | --- |
| Mapping | Selected node and parameter mappings are present, supported, and verified. |
| Builder | Catalog availability, materialization, inspector behavior, connection candidates, and profile switching are correct. |
| Static validation | The complete graph validates against the exact profile, including shapes and outputs. |
| Source | Generated source names the exact selected API/backend and has no silent fallback. |
| Functional validation | The selected worker succeeds, or returns the explicit expected unavailable diagnostic. |
| Export | Capability gating, submission, artifact verification, and reload/forward check match the profile. |
| Persistence | Save/reopen preserves graph, parameters, profile, outputs, and invalid repair state. |

## Carry-forward evidence, not closure

These reports provide useful starting points but do not close the full ledger:

| Existing evidence | What it covers | Carry-forward treatment |
| --- | --- | --- |
| node-functional-matrix-20260921.md | K/PyTorch CNN 1D/2D/3D, N/PyTorch CNN 2D, and six Activation values through Builder, save, and functional validation. | PASS only for those named cases at the recorded revision; all other nodes/profiles remain PENDING. |
| backend-compatibility-consolidation.md | Static six-profile mapping/parameter/connection contracts, repair retention, compiler/export boundaries, and focused browser evidence. | Use as static contract evidence; re-run live Builder/runtime rows on the current checkout. |
| live-backend-compatibility-e2e-20260913.md | A historical simple Dense graph across all six profiles plus selected restrictions and exports. | STALE for a current full-catalog claim; use as a revalidation map. |
| project_status_ledger.md | Current operational status, known blockers, and validation debt. | Inspect before each run and update only after new evidence. |

Current boundaries to preserve:

- Functional Validation is a deterministic, initialized, untrained forward check; it is not training or a real-user prediction workload.
- Native TensorFlow XLA and native PyTorch torch.compile are separate environment/toolchain gates; an unavailable compiler is BLOCKED or an explicit runtime rejection, never a hidden Keras success.
- Unsupported mappings and unsupported parameter values remain discoverable/repairable with typed diagnostics; they are not counted as successful execution.
- layer.reshape retains a source-export limitation in the current MVP boundary.
- Output adapters are optional semantic/task nodes; a valid graph may terminate at a value-producing computational node.

## Coverage levels and closure rules

| Level | Ledger unit | Minimum closure rule |
| --- | --- | --- |
| L0 | Catalog inventory | Live catalog captured; every definition/version appears exactly once. |
| L1 | Single-node behavior | Every definition is added/materialized or correctly disabled for each profile; defaults, inspector, reset, and a structurally valid fixture are covered. |
| L2 | Parameter behavior | Every declared parameter has default, valid, boundary/nullable, conditional, and profile-capability checks where applicable. |
| L3 | Port/edge behavior | Every declared compatible output-to-input edge candidate is checked in a valid shape; representative incompatible pairs cover each diagnostic reason. |
| L4 | Graph composition | Every node appears in at least one valid graph for each profile where its mapping is supported; special-port and shape-changing nodes have dedicated graphs. |
| L5 | Runtime/profile | Each supported profile has Builder creation, validation, source, functional worker, profile-switch, and applicable export evidence. |
| L6 | Persistence/recovery | Save/reopen, revision freshness, invalid repair, unsupported-profile retention, and failure cleanup are recorded. |

“All combinations” means the generated L3 candidate-edge set plus the L4 graph-family set is closed. It does not require enumerating every arbitrary graph of unbounded depth. A node with multiple ports, conditional ports, state outputs, variadic inputs, or backend-specific parameters must receive a dedicated L4 case.

## Node roster seed

Use this roster to seed the per-run node sheet, then reconcile it against /api/nodes/catalog before testing. The ID is the stable ledger key; display names are only for Builder selection.

### Inputs — 8

- input.constant — Constant
- input.image — Image Input
- input.tensor — Input Tensor
- input.mask — Mask Input
- input.parameter — Parameter
- input.sequence — Sequence Input
- input.tabular — Tabular Input
- input.token — Token Input

### Core — 14

- layer.activation — Activation
- layer.batch_norm — Batch Normalization
- layer.dense — Dense / Linear
- layer.dropout — Dropout
- layer.embedding — Embedding
- layer.flatten — Flatten
- layer.group_norm — Group Normalization
- layer.layer_norm — Layer Normalization
- layer.prelu — Parametric ReLU
- layer.permute — Permute
- layer.rms_norm — RMS Normalization
- layer.repeat — Repeat
- layer.reshape — Reshape
- layer.unit_norm — Unit Normalization

### Convolutional — 18

- conv.average_pool1d — Average Pooling 1D
- conv.average_pool2d — Average Pooling 2D
- conv.average_pool3d — Average Pooling 3D
- conv.conv1d — Conv1D
- conv.conv2d — Conv2D
- conv.conv3d — Conv3D
- conv.cropping — Cropping
- conv.depthwise — Depthwise Convolution
- conv.global_pool1d — Global Pooling 1D
- conv.global_pool2d — Global Pooling 2D
- conv.global_pool3d — Global Pooling 3D
- conv.max_pool1d — Max Pooling 1D
- conv.max_pool2d — Max Pooling 2D
- conv.max_pool3d — Max Pooling 3D
- conv.padding — Padding
- conv.separable — Separable Convolution
- conv.transposed — Transposed Convolution
- conv.upsampling — Upsampling

### Recurrent — 5

- recurrent.bidirectional — Bidirectional Recurrent Layer
- recurrent.gru — GRU
- recurrent.lstm — LSTM
- recurrent.simple_rnn — Simple RNN
- recurrent.stack — Recurrent Stack

### Transformer — 10

- attention.cross — Cross-attention
- attention.multi_head — Multi-head Attention
- attention.self — Self-attention
- transformer.attention_mask — Attention Mask
- transformer.decoder_block — Transformer Decoder Block
- transformer.encoder_block — Transformer Encoder Block
- transformer.feed_forward — Feed-forward Transformer Block
- transformer.positional_encoding — Positional Encoding
- transformer.residual — Residual Connection
- transformer.stack — Complete Transformer Stack

### Merge — 14

- merge.add — Add
- merge.average — Average
- merge.concatenate — Concatenate
- merge.divide — Divide
- merge.dot — Dot Product
- merge.matmul — Matrix Multiplication
- merge.maximum — Maximum
- merge.minimum — Minimum
- merge.multiply — Multiply
- merge.residual — Residual Connection
- merge.skip — Skip Connection
- merge.split — Split
- merge.stack — Stack
- merge.subtract — Subtract

### Mathematical — 73

- math.abs — Absolute Value
- math.add — Add
- math.all — All
- math.any — Any
- math.argmax — Argmax
- math.argmin — Argmin
- math.cast — Cast
- math.ceil — Ceiling
- math.clip — Clip
- math.concatenate — Concatenate
- math.cos — Cosine
- math.divide — Divide
- math.dot — Dot Product
- math.equal — Equal
- math.erf — Error Function
- math.expand_dims — Expand Dimensions
- math.exp — Exponential
- math.flatten — Flatten
- math.floor — Floor
- math.gelu — GELU
- math.greater — Greater Than
- math.greater_equal — Greater or Equal
- math.cosh — Hyperbolic Cosine
- math.sinh — Hyperbolic Sine
- math.tanh — Hyperbolic Tangent
- math.less — Less Than
- math.less_equal — Less or Equal
- math.logsumexp — Log-Sum-Exp
- math.log — Logarithm
- math.log1p — Logarithm 1p
- math.logical_and — Logical AND
- math.logical_or — Logical OR
- math.logical_xor — Logical XOR
- math.matmul — Matrix Multiplication
- math.maximum — Maximum
- math.max — Maximum Reduction
- math.mean — Mean
- math.minimum — Minimum
- math.min — Minimum Reduction
- math.multiply — Multiply
- math.negative — Negative
- math.not_equal — Not Equal
- math.power — Power
- math.prod — Product Reduction
- math.ravel — Ravel
- math.relu — ReLU
- math.reciprocal — Reciprocal
- math.rsqrt — Reciprocal Square Root
- math.reduce — Reduction Operations
- math.repeat — Repeat
- math.reshape — Reshape
- math.round — Round
- math.sqrt — Square Root
- math.squeeze — Squeeze Dimensions
- math.stack — Stack
- math.stddev — Standard Deviation
- math.stop_gradient — Stop Gradient
- math.subtract — Subtract
- math.sum — Sum
- math.take — Take
- math.tan — Tangent
- math.transpose — Transpose
- math.variance — Variance
- math.expression — Safe Mathematical Expression
- math.sigmoid — Sigmoid
- math.softmax — Softmax
- math.sign — Sign
- math.sin — Sine
- math.softplus — Softplus
- math.square — Square
- math.slice — Tensor Slicing
- math.index — Tensor Indexing
- math.tile — Tile

### Outputs — 7

- output.classification — Classification Output
- output.custom — Custom Output Tensor
- output.detection — Object Detection Output
- output.multi_head — Multi-output Head
- output.regression — Regression Output
- output.segmentation — Segmentation Output
- output.sequence — Sequence Output

## Ledger A — per-node and parameter sheet

Create one row for every node/profile pair from the live catalog. Keep expected capability status separate from observed runtime status.

| Node ID | Category | Fixture family | K/Tf | K/PT | K/JAX | N/Tf | N/PT | N/JAX | Parameters | Composition | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one row per live manifest |  |  | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |

For each supported node/profile row, the minimum Builder/runtime sequence is:

1. find the catalog item and confirm its enabled/disabled state and reason;
2. materialize the node through the user-facing Builder path;
3. inspect defaults, edit a representative value, reset, and save/reopen;
4. connect it in its fixture family and confirm server connection validation;
5. run complete static validation and confirm inferred shapes/output bindings;
6. compile source for the exact profile when the exporter contract supports it;
7. run Functional Validation and confirm worker profile/API, state, metadata, and cleanup;
8. record console/page errors, screenshots, request/job IDs, and artifact links.

For EXPECTED_UNSUPPORTED or BLOCKED rows, perform the same catalog/profile setup and capture the explicit reason and fail-closed behavior. Do not attempt to make an unsupported row green by switching profiles mid-row.

### Parameter sub-ledger

Every node parameter needs a row or a linked parameter batch with:

| Node ID | Parameter ID | Default | Valid boundary/nullable | Enum/options | Conditional state | Profile capability | Invalid diagnostic | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one row per declared parameter |  | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |

At minimum, cover these interaction groups:

| Group | Required combinations |
| --- | --- |
| Convolution/pooling | rank, filters/size, kernel, stride, padding, dilation, data format, channel transition, and spatial boundary behavior. |
| Core shape layers | flatten/reshape/permute/repeat with valid and impossible targets, dynamic dimensions, and source-export boundary. |
| Recurrent | rank-three input, units, return_sequences, return_state, state ports, dropout/recurrent dropout, stateful/backwards/unroll, and native restrictions. |
| Attention/transformer | rank-three inputs, embedding/head divisibility, masks, self/cross inputs, residual paths, stack depth, and conditional parameters. |
| Merge and tensor-list | equal-shape fan-in, concatenation axis, variadic inputs, split outputs, residual/skip shape equality, and multi-output selection. |
| Mathematical | unary/binary/reduction/tensor families, rank changes, dtype changes, indexing/slicing, safe-expression allowlist, and invalid expressions. |
| Output/task contracts | classification/regression/segmentation/sequence/multi-head semantics, activation/loss compatibility, and inferred versus explicit output bindings. |

## Ledger B — exhaustive port/edge combinations

Build this sheet from every output port × input port in the frozen catalog, not from a hand-picked list. Group candidates by port kind, rank, tensor-list element kind, required/enabled state, cardinality, and shape fixture.

| Edge ID | Source node/port | Target node/port | Shape fixture | Profile | Positive connection | Static graph validation | Runtime composition | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated from frozen catalog |  |  |  | K/Tf, K/PT, K/JAX, N/Tf, N/PT, N/JAX | PENDING | PENDING | PENDING |  |

Close the edge sheet only when:

- every compatible candidate edge has one valid-shape positive result for every profile where both mappings are supported;
- disabled/unavailable ports are rejected with the selected profile’s reason;
- representative incompatible pairs cover wrong kind, wrong rank, tensor versus tensor-list, mask/scalar mismatch, duplicate occupancy, disabled port, and multiple-input/cardinality errors;
- cycle, self-loop, dangling reference, unreachable component, and input/output contribution rules have evidence;
- fan-out, fan-in, multi-output/state ports, explicit output bindings, and inferred outputs are represented.

## Ledger C — graph composition families

Use stable family IDs so the same graph intent can be compared across profiles and future revisions. Each family must record the exact node IDs, parameters, shapes, edges, outputs, and profile.

| Family ID | Composition | Required focus | Current seed evidence | Future status |
| --- | --- | --- | --- | --- |
| GF-001 | Input → Dense/Activation → output | canonical linear path, activation options, output semantics | Existing representative Builder evidence | PENDING |
| GF-002 | 1D input → Conv1D → pooling → global pooling → output | rank-3 shape and 1D spatial transitions | node-functional-matrix-20260921.md | PASS only for named K/PyTorch case; expand |
| GF-003 | 2D image → Conv2D → pooling → global pooling → output | rank-4 shape, channels, 2D transitions | node-functional-matrix-20260921.md | PASS only for named K/PyTorch and N/PyTorch cases; expand |
| GF-004 | 3D tensor → Conv3D → pooling → global pooling → output | rank-5 shape and 3D transitions | node-functional-matrix-20260921.md | PASS only for named K/PyTorch case; expand |
| GF-005 | Sequence → Simple RNN/GRU/LSTM → sequence output | rank-three input and return_sequences | Current NW-T3-01 and NW-T3-02 recurrent cases | PASS for the current K/PyTorch Simple RNN, GRU, and LSTM cases; broader parameters and profiles remain pending |
| GF-006 | Recurrent node with return_state → multi-output/state consumers | state ports, output selection, serialization | Current NW-T3-01 and NW-T3-02 recurrent state-consumer cases | PASS for the current K/PyTorch Simple RNN, GRU, and LSTM state cases; full multi-output/profile matrix remains pending |
| GF-007 | Sequence → self/multi-head attention → output | head divisibility, attention axes, output rank | Static contract evidence | PENDING live Builder/runtime |
| GF-008 | Query/key/value or mask → cross-attention/transformer block | multi-input ports, masks, residuals, conditional parameters | Static contract evidence | PENDING live Builder/runtime |
| GF-009 | Two branches → merge/concatenate/stack/residual/skip → output | fan-in, shape equality, axis, tensor-list cardinality | NW-T3-01 Concatenate, NW-T3-08 arithmetic fan-in, and NW-T3-09 structural merge cases | PASS for the selected K/PyTorch and native-PyTorch Concatenate, Stack, Split, Dot, MatMul, Residual, and Skip paths; expand merge axis/cardinality/profile matrix |
| GF-010 | Input → mathematical unary/binary/reduction/tensor operations | dtype/rank/shape rules and safe expressions | Native parity supporting evidence; current NW-T3-08 arithmetic cases | PASS for the current K/PyTorch arithmetic and Concatenate paths plus native Add/Average neighbors; full mathematical family remains pending |
| GF-011 | Branch and fan-out graph | one output to multiple compatible consumers | Existing graph validation coverage | PENDING per profile |
| GF-012 | Multi-input and multi-output graph | explicit output bindings, stable port IDs, independent terminals | Native parity supporting evidence | PENDING Builder/profile coverage |
| GF-013 | Output adapter variants | classification, regression, segmentation, sequence, multi-head, custom, detection | Static/output contract evidence | PENDING live profile coverage |
| GF-014 | Invalid graph → repair → valid graph | invalid parameter, unsupported node/value, cycle, shape mismatch, profile switch retention | Representative Builder investigation | PASS only for covered cases; expand |
| GF-015 | Save → reopen → profile switch → revalidate | revision, stale validation, retained incompatible state, no pruning/fallback | Representative Builder investigation | PENDING all six profiles |

For each family, repeat the graph over every profile that the frozen catalog marks supported. If the family is not supported by a profile, record EXPECTED_UNSUPPORTED or BLOCKED with the exact mapping/runtime reason.

## Ledger D — backend/profile execution and export

This is the profile-level summary after the node and family sheets are populated.

| Profile | Catalog/mapping | Builder creation | Static validation | Source exactness | Functional worker | Export capability/artifact | Profile switching | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K/TensorFlow | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |
| K/PyTorch | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |
| K/JAX | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |
| N/TensorFlow | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |
| N/PyTorch | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |
| N/JAX | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |  |

The profile summary may be PASS only when every supported L1/L3/L4 row in that profile is closed and all unavailable rows have explicit expected/blocker evidence. A small representative graph cannot close full-catalog profile coverage.

## Evidence contract

For every PASS or EXPECTED_UNSUPPORTED row, retain:

- run ID, repository revision, profile, catalog fingerprint, environment fingerprint, and graph revision;
- the test/browser command or manual workflow used;
- Builder screenshot or browser trace for user-facing interactions;
- validation response and diagnostic payload;
- generated source or source hash where source is in scope;
- functional job ID, final state, selected worker/API/backend, input/output metadata, and untrained boundary;
- export capability response and artifact verification where export is in scope;
- browser pageerror/console.error result and backend log correlation;
- cleanup result, including stopped task-owned processes and test data.

Store new reports, screenshots, traces, logs, and matrices under assets/QA. Link them from this ledger and from the project status ledger when a component status changes.

## Current bounded run: NW-T3-01

Date: 2026-09-27
Repository: `develop` at implementation commit `09d53a0`
Run report: [NW-T3-01 risk-based catalog slice](comprehensive-validation-20260927/NW-T3-01.md)
Catalog: 149 nodes; fingerprint `e0ea2f09ef44a4b3fcb10988495fcdea578f6d504c9bc39915e224d50276f439`
Profile: `execution_backend=pytorch`, `keras_enabled=true`, Keras API / PyTorch backend, CPU worker
Evidence root: `assets/QA/comprehensive-validation-20260927/`

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | Live catalog count, category counts, IDs, and fingerprint are frozen; this does not close node-level coverage. |
| L1 single-node behavior | PASS for `Simple RNN` and `Concatenate` materialization | The remaining 147 definitions and profile-specific enable/disable behavior remain PENDING. |
| L2 parameter behavior | PASS for recurrent `units` / `return_sequences` / `return_state` and Concatenate `axis=-1` | Other parameters, boundaries, nullable values, and profile capabilities remain PENDING. |
| L3 port/edge behavior | PASS for recurrent sequence/state consumers and two-input tensor-list fan-in | Exhaustive catalog edge generation and incompatible-pair coverage remain PENDING. |
| L4 graph composition | PASS for GF-005, GF-006, GF-009, and the selected GF-010 Concatenate case | Each result is a named K/PyTorch graph only; the family/profile matrix remains PENDING. |
| L5 runtime/profile | PASS for Functional Validation on the selected K/PyTorch lane | Source, export, profile switching, and the other five profiles remain PENDING. |
| L6 persistence/recovery | PASS for save/reopen at revision 2 in both cases | Invalid repair, profile switching, export reload, and failure cleanup across the matrix remain PENDING. |

The run found and fixed a static parameter-count defect: dedicated `recurrent.simple_rnn` and `recurrent.gru` definitions were using the LSTM multiplier when their `cell` parameter was absent. The regression and final live evidence are linked from the [run report](comprehensive-validation-20260927/NW-T3-01.md). This is a bounded PASS, not full Model Builder certification.

## Current bounded run: NW-T3-02

Date: 2026-09-28
Repository: `develop` at implementation commit `4ec01814ca1a3d53a83d1012c84181f92fc000e4`
Run report: [NW-T3-02 recurrent GRU/LSTM variant slice](comprehensive-validation-20260928/NW-T3-02.md)
Run manifest: [NW-T3-02-run-manifest.json](comprehensive-validation-20260928/NW-T3-02-run-manifest.json)
Catalog: 149 nodes; fingerprint `e0ea2f09ef44a4b3fcb10988495fcdea578f6d504c9bc39915e224d50276f439`; definition IDs match the prior freeze
Profile: `execution_backend=pytorch`, `keras_enabled=true`, Keras API / PyTorch backend, CPU worker
Evidence root: `assets/QA/comprehensive-validation-20260928/`

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | Live count, category counts, IDs, and fingerprint were checked against the prior freeze; node-level coverage remains open. |
| L1 single-node behavior | PASS for Simple RNN, GRU, LSTM, and Concatenate materialization | 145 definitions and profile-specific enable/disable behavior remain PENDING. |
| L2 parameter behavior | PASS for recurrent units, `return_sequences`, `return_state`, GRU `reset_after`, and Concatenate `axis=-1` | Other parameters, constraints, nullable values, and profile capabilities remain PENDING. |
| L3 port/edge behavior | PASS for recurrent sequence/state consumers and selected tensor-list fan-in | Exhaustive catalog edge generation and incompatible-pair coverage remain PENDING. |
| L4 graph composition | PASS for GF-005, GF-006, GF-009, and the selected GF-010 Concatenate case | Full graph-family and profile matrix remains PENDING. |
| L5 runtime/profile | PASS for Functional Validation on the selected K/PyTorch CPU lane | Source, export, profile switching, and the other five profiles remain PENDING. |
| L6 persistence/recovery | PASS for save/reopen at revision 2 in the exercised GRU/LSTM and neighbor cases | Invalid repair, profile switching, export reload, and matrix-wide failure cleanup remain PENDING. |

This run found and fixed a live static/runtime parameter-count mismatch: the
first GRU/LSTM run reported static total `672` while the successful
Keras/PyTorch worker reported `696`. Keras GRU `reset_after=true` uses two bias
vectors, so `shape_inference.py` now counts two GRU bias vectors by default and
one when `reset_after=false`. Regression coverage records GRU `312`/`288`,
Simple RNN `96`, and LSTM `384`. The final graph persisted at revision 2,
returned five outputs, completed successfully on CPU, and emitted no browser
errors. This is a bounded PASS, not full Model Builder certification.

## Current run: recurrent wrapper continuation

Date: 2026-09-30
Repository: `develop`; source baseline `416c158c61349d7a1106e71e64fba4174d3178a5` before scoped campaign changes
Run report: [recurrent wrapper continuation](comprehensive-validation-20260930/NW-T3-recurrent-wrapper-continuation.md)
Catalog: 149 nodes; fingerprint `acf538fb04f50552e1577299aa3f0cc09511d4e7d0937f73aa8418125a3a9ef8`
Profile: `execution_backend=pytorch`, `keras_enabled=true`, Keras API / PyTorch backend, CPU worker
Evidence root: `assets/QA/comprehensive-validation-20260930/`

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | The live catalog count and fingerprint match the freeze; node-level coverage remains open. |
| L1 single-node behavior | PASS for Bidirectional Recurrent Layer, Recurrent Stack, and retained Simple RNN/GRU/LSTM/Concatenate neighbors | The remaining definitions and profile-specific enable/disable behavior remain PENDING. |
| L2 parameter behavior | PASS | All bidirectional merge shapes, standard LSTM counts, stack layers, and invalid-layer diagnostics pass. The acceptance criterion now uses standard LSTM semantics: observed static/runtime counts are `256`/`272`; the obsolete `64`/`68` expectation is removed. |
| L3 port/edge behavior | PASS for the selected sequence input/wrapper/output graphs | Exhaustive catalog edge generation and incompatible-pair coverage remain PENDING. |
| L4 graph composition | PASS for the selected bidirectional and two-layer stack graphs | Full graph-family and profile composition remain PENDING. |
| L5 runtime/profile | PASS for the selected Keras/PyTorch CPU lane | Source/export, profile switching, the other five profiles, and positive PyTorch Inductor remain PENDING/PARTIAL at their existing boundaries. |
| L6 persistence/recovery | PASS for blocked stack repair, save/reopen at revision 2, validation, runtime, and fixture cleanup | Matrix-wide recovery and failure cleanup remain PENDING. |

The corrected serial authoritative catalog run passed `6/6` cases and captured
rendered Builder/runtime screenshots plus empty browser-error payloads. No
implementation change was made to force the old counts. The historical
2026-09-28 report remains retained for provenance; this 2026-09-30 run is the
active acceptance evidence.

## Current bounded run: NW-T3-03 revalidation

Date: 2026-09-29
Repository: develop at current checkout HEAD `42666a8374599e27367fcc2c9ae2a1ac77798ee7` with scoped validation changes
Run report: [NW-T3-03 current-checkout revalidation](comprehensive-validation-20260929/NW-T3-03-revalidation.md)
Run manifest: [NW-T3-03 revalidation manifest](comprehensive-validation-20260929/NW-T3-03-revalidation-run-manifest.json)
Catalog: 149 nodes; fingerprint `306042a67a66852161de34e2c72c35bccb509a80b1245172af11f53180b500d1`; all definition IDs are in the shared freeze
Profile: selected Keras API / PyTorch backend / CPU and native PyTorch / CPU neighbors
Evidence root: assets/QA/comprehensive-validation-20260929/

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | Live catalog, category counts, metadata, and current fingerprint were frozen; node-level coverage remains open. |
| L1 single-node behavior | PASS for Conv1D, Average Pooling 1D, Global Pooling 1D, Conv2D, Max Pooling 2D, Global Pooling 2D, Conv3D, Average Pooling 3D, and Global Pooling 3D | The other nine convolutional definitions and 140 non-convolutional definitions remain PENDING. |
| L2 parameter behavior | PASS for the defaults and shape-changing parameters exercised by the four named graphs | Full convolution/pooling parameter boundaries, nullable and conditional states, and profile capabilities remain PENDING. |
| L3 port/edge behavior | PASS for the four selected input-to-convolution-to-pooling chains | Exhaustive catalog edge generation and incompatible-pair coverage remain PENDING. |
| L4 graph composition | PASS for GF-002, GF-003, and GF-004 in the named Keras/PyTorch cases plus the native PyTorch Conv2D neighbor | The remaining graph families and profile matrix remain PENDING. |
| L5 runtime/profile | PASS for Functional Validation on the selected Keras/PyTorch CPU and native PyTorch CPU lanes | Source, export, profile switching, the other four profiles, and training remain PENDING. |
| L6 persistence/recovery | PASS for save/reload of all four graphs, browser-error checks, and task-fixture cleanup | Invalid repair, retained incompatible state, profile switching, export reload, and matrix-wide recovery remain PENDING. |

The authoritative unchanged browser matrix passed 4/4 cases with one worker on
the dedicated fresh resource root. Static shapes, parameter counts, exact
worker identity, runtime outputs, save/reload assertions, and empty
browser-error payloads passed for every case. No convolution product defect was
found or changed. This remains a bounded PASS, not full Model Builder
certification.

## Current bounded run: NW-T3-04

Date: 2026-09-28
Repository: develop at implementation revision `65f1f7e99ec059876efb0fba256738ea7fb2ec01`
Run report: [NW-T3-04 advanced and spatial convolution](comprehensive-validation-20260928/NW-T3-04.md)
Run manifest: [NW-T3-04 run manifest](comprehensive-validation-20260928/NW-T3-04-run-manifest.json)
Catalog: 149 nodes; current live-response fingerprint `ebff9fdbaf3d2a8ff09d783c6315fa81434362f5cd266b6b1b6b7fb589a3f333`; nine remaining advanced/spatial convolution definition IDs were confirmed live and Keras/PyTorch-supported
Profile: selected Keras API / PyTorch backend / CPU and native PyTorch / CPU neighbor for depthwise convolution
Evidence root: assets/QA/comprehensive-validation-20260928/

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | Live catalog count, schema, selected IDs, profile mappings, and readiness were captured; full catalog coverage remains open. |
| L1 single-node behavior | PASS for the nine selected advanced/spatial definitions | The other 140 definitions and untested parameter states remain pending. |
| L2 parameter behavior | PASS for the live defaults and exercised shape-changing parameters | Full nullable, conditional, boundary, and profile-specific parameter coverage remains pending. |
| L3 port/edge behavior | PASS for the selected input-to-operation-to-output edges | Exhaustive compatible and incompatible edge coverage remains pending. |
| L4 graph composition | PASS for ten three-node graphs | Larger graph families and cross-family composition remain pending. |
| L5 runtime/profile | PASS for Keras/PyTorch CPU and native PyTorch CPU; browser matrix 11/11 | Other live profiles, source/export, training, and compiler-positive paths remain pending. |
| L6 persistence/recovery | PASS for save/reload, browser errors, project deletion, process cleanup, and port release | Invalid repair, retained incompatible state, profile switching, export reload, and matrix-wide recovery remain pending. |

The authoritative serial browser matrix passed 11/11 cases. The cases covered Max Pooling 1D, Average Pooling 2D, Max Pooling 3D, Depthwise, Separable, and Transposed Convolution, Upsampling, Padding, Cropping, and a native-PyTorch Depthwise neighbor. Every case produced the expected static shape and parameter count, saved with three nodes and two edges, reloaded with three nodes, completed with the exact selected worker, and emitted an empty browser error payload. The in-app Browser observation confirmed the rendered live convolutional catalog. Full Ruff is green after three narrow import-block repairs; strict Pyright remains PARTIAL at 58 existing errors. The existing recurrent-wrapper, torch.compile, and 1/2/4-worker matrix limitations remain unchanged. The next selected catalog slice was NW-T3-07 Transformer blocks; that bounded run is recorded below. This remains a bounded PASS, not full Model Builder certification.

## Current bounded run: NW-T3-05

Date: 2026-09-28
Repository: develop; validation began at implementation revision `492567e06472b2b5bafd157135bec4099f3e3989`
Run report: [NW-T3-05 recurrent nodes](comprehensive-validation-20260928/NW-T3-05.md)
Run manifest: [NW-T3-05 run manifest](comprehensive-validation-20260928/NW-T3-05-run-manifest.json)
Catalog: 149 nodes; live recurrent category contains five definitions and all five expose Keras/PyTorch mappings
Profile: selected Keras/PyTorch CPU and native PyTorch CPU; TensorFlow/JAX recurrent runtime and remaining profiles are outside this slice
Evidence root: assets/QA/comprehensive-validation-20260928/

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | Live catalog and recurrent IDs/mappings were frozen; full catalog coverage remains open. |
| L1 single-node behavior | PASS for Bidirectional, GRU, LSTM, Recurrent Stack, and Simple RNN | Remaining definitions and untested recurrent parameter states remain pending. |
| L2 parameter behavior | PASS for selected units, sequence output, cell, merge mode, and stack layer values | Full conditional/boundary/profile-specific parameter coverage remains pending; the separate Keras wrapper count mismatch remains PARTIAL. |
| L3 port/edge behavior | PASS for selected sequence input-to-node-to-output chains | Exhaustive recurrent compatible/incompatible edge coverage remains pending. |
| L4 graph composition | PASS for the six existing Keras/PyTorch recurrent cases and five native PyTorch CPU graphs | Larger, cross-family, and all remaining graph families remain pending. |
| L5 runtime/profile | PASS for the selected Keras/PyTorch CPU and native PyTorch CPU lanes | TensorFlow/JAX recurrent runtime, source/export, training, and compiler-positive paths remain pending. |
| L6 persistence/recovery | PASS for validation, save/reload, exact worker identity, browser-error checks, project deletion, and process/port cleanup | Matrix-wide recovery and full profile/export closure remain pending. |

The current authoritative browser matrix passed 6/6 native cases and 6/6
existing Keras/PyTorch recurrent cases. A repository-wide backend rerun passed
after fixing a compiler-gate alias defect: worker probes use
`tensorflow_jit`/`torch_compile`, while the controller previously checked only
`tensorflow_function`/`pytorch_compile`. The controller now uses the existing
alias-aware capability resolver. Full Ruff, frontend unit/lint/build, and the
focused backend/catalog/export/parity suite pass. Strict Pyright remains
PARTIAL at 58 existing errors. NW-T3-05 remains a bounded PASS, not full Model
Builder certification.

## Current bounded run: NW-T3-06 attention primitives

Date: 2026-09-29
Repository: develop at current checkout HEAD `42666a8374599e27367fcc2c9ae2a1ac77798ee7` with scoped validation changes
Run report: [NW-T3-06 attention primitives](comprehensive-validation-20260929/NW-T3-06.md)
Run manifest: [NW-T3-06 run manifest](comprehensive-validation-20260929/NW-T3-06-run-manifest.json)
Catalog: 149 nodes; attention IDs `attention.self`, `attention.cross`, and `attention.multi_head`; fingerprint `306042a67a66852161de34e2c72c35bccb509a80b1245172af11f53180b500d1`
Profile: Keras/PyTorch CPU and native PyTorch CPU browser lanes; focused native parity across available TensorFlow, PyTorch, and JAX backends
Evidence root: assets/QA/comprehensive-validation-20260929/

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | Attention IDs, public parameters, profile capabilities, runtime versions, category counts, all definition IDs, and live fingerprint were frozen; full catalog coverage remains open. |
| L1 single-node behavior | PASS for self, cross, and multi-head attention | Other Transformer definitions and untested attention parameter states remain pending. |
| L2 parameter behavior | PASS for heads, key/value dimensions, explicit output shape, bias count, native fixed capability values, and invalid layers repair | Full parameter boundary and all profile-specific runtime coverage remain pending. |
| L3 port/edge behavior | PASS for self Query, cross Query/Context, and output connections through the real Builder | Exhaustive attention edge compatibility and larger compositions remain pending. |
| L4 graph composition | PASS for the three representative attention graph forms | Cross-family Transformer composition and larger graph families remain pending. |
| L5 runtime/profile | PASS for Keras/PyTorch CPU and native PyTorch CPU browser cases; focused parity passes for available native TensorFlow/PyTorch/JAX | Native TensorFlow/JAX browser execution, all six live profile/export paths, and training remain pending. |
| L6 persistence/recovery | PASS for save/reload and invalid `layers=2` to valid `layers=1` repair, including runtime success and cleanup | Matrix-wide recovery, retained incompatible-state coverage, export reload, and accessibility remain pending. |

The serial dedicated attention browser slice passed 7/7; the repaired recovery
case was rerun independently and passed 1/1. Full strict Pyright remains
PARTIAL at 58 existing errors. This remains a bounded PASS, not full Model
Builder certification.

## Current bounded run: NW-T3-07 Transformer blocks

Date: 2026-09-29
Repository: develop; validation source tree was based on `f1e03831bdcf36037c3c194022981cd017264dd2`
Run report: [NW-T3-07 Transformer blocks](comprehensive-validation-20260929/NW-T3-07.md)
Run manifest: [NW-T3-07 run manifest](comprehensive-validation-20260929/NW-T3-07-run-manifest.json)
Catalog: 149 nodes; Transformer category contains ten entries; seven remaining Transformer definitions were frozen with catalog fingerprint `306042a67a66852161de34e2c72c35bccb509a80b1245172af11f53180b500d1`
Profile: selected Keras/PyTorch CPU and native PyTorch CPU browser lanes; focused validation/parity across the available TensorFlow, PyTorch, and JAX lanes
Evidence root: assets/QA/comprehensive-validation-20260929/evidence/nw-t3-07-transformer/

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | All seven remaining Transformer IDs, declared mappings, live count, category, and fingerprint were frozen; full catalog coverage remains open. |
| L1 single-node behavior | PASS for encoder, decoder, feed-forward, stack, positional encoding, and residual; mask input materialization PASS | Exhaustive Transformer parameter states and standalone mask execution remain pending. |
| L2 parameter behavior | PASS for selected head/key/layer/FFN/dropout/pre-normalization values and exact counts | Full conditional and profile-specific parameter coverage remains pending. |
| L3 port/edge behavior | PASS for selected sequence, Query/Context, residual fan-in, and output edges; typed mask-to-tensor rejection PASS | No mask-capable terminal consumer is present, and exhaustive compatible/incompatible edge coverage remains pending. |
| L4 graph composition | PASS for seven selected three- or four-node graphs | Larger cross-family and remaining graph families remain pending. |
| L5 runtime/profile | PASS for six Keras/PyTorch CPU cases and one native PyTorch CPU encoder; focused parity passes | TensorFlow/JAX live Transformer execution, all-six profile/export closure, and standalone mask execution remain pending. |
| L6 persistence/recovery | PASS for save/reload, fail-closed `layers=0` UI state, repair to `layers=1`, exact runtime, and cleanup | Matrix-wide recovery, retained incompatible-state coverage, export reload, and accessibility remain pending. |

The serial Transformer browser run passed 9/9 catalog, positive, native, and
mask-boundary cases; the repaired invalid-layer case passed 1/1 after the
spec was corrected to assert the Builder's intended client-side invalid-value
guard. Focused backend validation passed 11 selected tests, native parity
passed 43 selected tests, and the two parameter-editor unit files passed 11
tests. The mask boundary remains explicitly `UNTESTED/PARTIAL` for standalone
runtime because the current catalog has no mask-capable terminal consumer.
Strict Pyright remains PARTIAL at 58 existing errors, and the worker matrix,
torch.compile, recurrent-wrapper count, and all-six-profile limitations remain
unchanged. NW-T3-07 is a bounded PASS, not full Model Builder certification.

## Current bounded run: NW-T3-08 Arithmetic merge and fan-in

Date: 2026-09-29
Repository: develop at validation revision `b9606f6c6a364c892c6b934be6c6aa6fba002646`
Run report: [NW-T3-08 arithmetic merge and fan-in](comprehensive-validation-20260929/NW-T3-08.md)
Run manifest: [NW-T3-08 run manifest](comprehensive-validation-20260929/NW-T3-08-run-manifest.json)
Catalog: 149 nodes; 14 merge entries; arithmetic definitions and live normalized catalog fingerprint `306042a67a66852161de34e2c72c35bccb509a80b1245172af11f53180b500d1`
Profile: selected Keras API / PyTorch backend / CPU and native PyTorch / CPU browser lanes
Evidence root: assets/QA/comprehensive-validation-20260929/evidence/nw-t3-08-merge/

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS for the live 149-node/14-merge catalog, seven named arithmetic IDs, mappings, and rendered Builder category | Full catalog coverage remains pending. |
| L1 single-node behavior | PASS for Add, Subtract, Multiply, Divide, Maximum, Minimum, and Average materialization | The remaining seven merge definitions remain pending. |
| L2 parameter behavior | PASS for the parameter-free arithmetic/average cases with zero parameters | Parameterized merge definitions and full profile-specific parameter states remain pending. |
| L3 port/edge behavior | PASS for three-input tensor-list fan-in, compatible shapes, and fail-closed incompatible `[4]`/`[5]` fan-in | Exhaustive merge cardinality, axis, rank, dtype, and edge compatibility remain pending. |
| L4 graph composition | PASS for nine selected positive graphs across Keras/PyTorch and native PyTorch | Larger cross-family graphs and remaining merge families remain pending. |
| L5 runtime/profile | PASS for seven Keras/PyTorch CPU and two native PyTorch CPU cases, all with exact workers and runtime shape `[1,4]` | TensorFlow/JAX browser execution, all-six profile/export closure, and positive Inductor remain pending. |
| L6 persistence/recovery | PASS for revision-2 save/reload, successful execution, invalid-state no-job behavior, project deletion, and process/port cleanup | Matrix-wide recovery, export reload, accessibility, and worker-matrix closure remain pending. |

The serial authoritative browser run passed 11/11. Every positive receipt
reported five nodes/four edges, static `[null,4]`, runtime `[1,4]`, zero
parameters, success, exact worker identity, and empty browser/request errors.
The negative receipt rejected the mismatched middle input with
`shape_inference_failed` and submitted zero execution jobs. Focused validation
passed 6 tests, native parity passed 15 selected tests, export regressions
passed 5 tests with an isolated basetemp, frontend unit/lint/build passed, and
canonical Ruff passed. Strict Pyright remains PARTIAL at 58 existing errors;
the worker matrix, torch.compile, recurrent-wrapper count, all-six-profile,
accessibility, and training limitations remain unchanged. The protected
isolated pytest directory is recorded as an environment cleanup limitation.
NW-T3-08 is a bounded PASS, not full Model Builder certification.

## Current bounded run: NW-T3-09 Structural merge catalog

Date: 2026-09-29
Repository: develop at the pre-commit validation checkout based on `b9dfb7a4a9f16fda2f0e0e762e815a822c3db0b6`
Run report: [NW-T3-09 structural merge catalog](comprehensive-validation-20260929/NW-T3-09.md)
Run manifest: [NW-T3-09 run manifest](comprehensive-validation-20260929/NW-T3-09-run-manifest.json)
Catalog: 149 nodes; 14 merge entries; live normalized catalog fingerprint `306042a67a66852161de34e2c72c35bccb509a80b1245172af11f53180b500d1`
Profile: selected Keras API / PyTorch backend / CPU and native PyTorch / CPU browser lanes
Evidence root: assets/QA/comprehensive-validation-20260929/evidence/nw-t3-09-structural-merge/

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS for all 149 live nodes, all 14 merge entries, structural operation contracts, verified Keras/PyTorch mappings, and rendered Merge category | Full catalog/profile freeze and exhaustive per-node closure remain pending. |
| L1 single-node behavior | PASS for Concatenate, Stack, Split, Dot, MatMul, Residual, and Skip materialization | Other definitions and broader parameter states remain pending. |
| L2 parameter behavior | PASS for Split `sections=3`/`axis=-1`, Concatenate default axis, and Stack default axis | Alternate axes, section/cardinality limits, dtype/rank boundaries, and profile-specific parameters remain pending. |
| L3 port/edge behavior | PASS for ordered binary operands, tensor-list fan-in, Split dynamic output IDs, selected multi-output binding, and invalid MatMul/Split rejection | Exhaustive port/edge combinations and all cardinality/rank/dtype boundaries remain pending. |
| L4 graph composition | PASS for the seven selected structural merge families on the named Keras/PyTorch and native PyTorch lanes | Larger cross-family graphs and the full family/profile matrix remain pending. |
| L5 runtime/profile | PASS for 13 positive graphs with exact workers and expected static/runtime shapes; two invalid graphs fail closed without jobs | TensorFlow/JAX browser execution, six-profile/export closure, and positive Inductor remain pending. |
| L6 persistence/recovery | PASS for revision-2 save/reload, runtime success, invalid no-job behavior, fixture deletion, and process/port cleanup | Matrix-wide recovery, structural source/export reload, accessibility, and worker-matrix closure remain pending. |

The final serial authoritative browser run passed 15/15: one catalog/render
case, 13 positive structural merge cases, and two invalid-shape cases. Split
materialized `output_0`/`output_1`/`output_2` for `sections=3`; Stack’s default
axis produced `[2,null,4]` statically and `[2,1,4]` at runtime. The run found
and fixed a real same-port operand-order defect in `ordered_incoming_edges`:
shared tensor-list edges now preserve graph connection order while named ports
retain manifest order. The focused regression and final browser Dot/MatMul
cases pass with the fix.

Focused validation passed 2 tests, split/merge execution passed 10 tests,
native merge parity passed 18 tests, catalog audit passed 6 tests, isolated
export regressions passed 5 tests, frontend unit/lint/build passed, and Ruff
passed. Strict Pyright remains PARTIAL at 58 existing errors; the worker
matrix, torch.compile, all-six-profile/export, accessibility, and training
limitations remain unchanged. NW-T3-09 is a bounded PASS, not full Model
Builder certification.

## Current bounded run: NW-T3-10 Mathematical operation catalog

Date: 2026-09-29
Repository: `develop`; validation checkout started at implementation revision `ff8ce1508e4d9bd0d5b0fa791398245375bad7ff` before the validation changes were committed
Run report: [NW-T3-10 mathematical operation catalog](comprehensive-validation-20260929/NW-T3-10.md)
Run manifest: [NW-T3-10 run manifest](comprehensive-validation-20260929/NW-T3-10-run-manifest.json)
Catalog: 149 nodes; 73 mathematical entries; live normalized catalog fingerprint `3d2cd10184858d906b904e8e9cee1a9aee715f6a70c79434891a4efa0b20a884`
Profiles: Keras API / PyTorch backend / CPU and native PyTorch / CPU browser lanes
Evidence root: `assets/QA/comprehensive-validation-20260929/evidence/nw-t3-10-mathematical-operations/`

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS for the live catalog, Mathematical category rendering, selected typed operation contracts, and implementation mappings | The remaining mathematical definitions and full catalog/profile freeze remain pending. |
| L1 single-node behavior | PASS for 13 selected mathematical definitions across both named CPU profiles | The remaining 60 mathematical definitions and alternate parameter states remain pending. |
| L2 parameter behavior | PASS for exponent, axes/axis, reduction operation/keepdims, cast dtype, transpose, expand-dims, expression, and slice/index parameters | Full parameter, rank, dtype, and edge boundary coverage remains pending. |
| L3 port/edge behavior | PASS for the selected unary/tensor paths, ordered binary operands, and output terminal | Exhaustive mathematical port/edge combinations and profile-specific compatibility remain pending. |
| L4 graph composition | PASS for 26 positive three-node/two-edge graphs with revision-2 save/reload | Larger cross-family graphs and the full family/profile matrix remain pending. |
| L5 runtime/profile | PASS for 26 exact-profile smoke-inference jobs; unsafe expression rejected before execution with zero jobs | Live TensorFlow/JAX Builder execution/export, all-six-profile closure, and positive Inductor remain pending. |
| L6 persistence/recovery | PASS for save/reload, receipts, empty browser/request diagnostics, fixture deletion, and process/port cleanup | Matrix-wide recovery, accessibility, source/export, and worker-matrix closure remain pending. |

The authoritative serial browser run passed 28/28: one catalog/render case,
26 positive profile cases, and one unsafe-expression case. The run found and
fixed three real issues: incomplete binary drafts were rejected before the
declared arity was satisfied; typed Keras/PyTorch reductions did not preserve
the manifest's `int64` output contract; and `math.transpose.axes` used a
shape-valued editor that rejected the valid axis index `0`. The final evidence
contains 54 JSON payloads and 54 PNG renders, and the project fixture list was
empty before shutdown.

Focused backend tests, the native TensorFlow/PyTorch/JAX operation matrix,
the argmin `int64` regression, targeted Ruff, changed-spec ESLint, and catalog
JSON parsing passed. Strict Pyright remains PARTIAL at 58 existing errors; the
worker matrix, torch.compile, recurrent-wrapper count, standalone Transformer
mask, all-six-profile/export, accessibility, and training limitations remain
unchanged. NW-T3-10 is a bounded PASS, not full Model Builder certification.

## Current bounded run: NW-T3-11/NW-T3-12 unary mathematical families

Date: 2026-09-29
Repository: `develop`; validation checkout before commit: `ef5c376626c8b64aeb3c0704d3ff452846561ecc`
Run report: [NW-T3-11/NW-T3-12 unary mathematical validation](comprehensive-validation-20260929/NW-T3-11-12.md)
Run manifest: [NW-T3-11/NW-T3-12 run manifest](comprehensive-validation-20260929/NW-T3-11-12-run-manifest.json)
Catalog: 149 nodes; 73 Mathematical entries; live normalized fingerprint `acf538fb04f50552e1577299aa3f0cc09511d4e7d0937f73aa8418125a3a9ef8`
Profiles: Keras API / PyTorch backend / CPU and native PyTorch / CPU browser lanes; native TensorFlow/PyTorch/JAX CPU operation matrix
Evidence root: `assets/QA/comprehensive-validation-20260929/evidence/nw-t3-11-12-unary-mathematics/`

The live crosswalk selected 17 unary shape-preserving definitions for NW-T3-11,
including the additional live `math.clip` definition, and seven hyperbolic or
activation/probability definitions for NW-T3-12. The 13 NW-T3-10 definitions
remain carry-forward evidence and were not repeated as primary cases. The
bounded run passed 48 positive browser cases, 72 native backend cases, exact
save/reload and cleanup, and all selected L0-L6 rows below. The result is a
bounded PASS, not full Mathematical catalog, six-profile, source/export,
training, accessibility, worker-matrix, or later-slice closure.

| Slice | Level | Result | Bounded evidence |
| --- | --- | --- | --- |
| NW-T3-11 | L0 catalog inventory | PASS | 17 live definitions, Mathematical Builder rendering, mappings, and fingerprint. |
| NW-T3-11 | L1 node behavior | PASS | 17 unary nodes materialized and runtime-validated on the two exact browser lanes. |
| NW-T3-11 | L2 parameter behavior | PASS | Unary contracts plus `math.clip` min/max parameters; exhaustive parameter/rank/edge coverage remains open. |
| NW-T3-11 | L3 ports/edges | PASS | Single input/output ports and normal Builder connections for all selected definitions. |
| NW-T3-11 | L4 composition | PASS | Input → operation → output graphs, static validation, revision-2 save/reload. |
| NW-T3-11 | L5 runtime/profile | PASS | 34 browser cases plus 51 native TensorFlow/PyTorch/JAX CPU cases; two browser profiles only. |
| NW-T3-11 | L6 persistence/recovery | PASS | Exact graph receipts, diagnostics, project deletion, process shutdown, and port release. |
| NW-T3-12 | L0 catalog inventory | PASS | Seven live definitions, Mathematical Builder rendering, mappings, and fingerprint. |
| NW-T3-12 | L1 node behavior | PASS | Seven unary nodes materialized and runtime-validated on the two exact browser lanes. |
| NW-T3-12 | L2 parameter behavior | PASS | Selected hyperbolic/activation/probability contracts; exhaustive parameter/rank/edge coverage remains open. |
| NW-T3-12 | L3 ports/edges | PASS | Single input/output ports and normal Builder connections for all selected definitions. |
| NW-T3-12 | L4 composition | PASS | Input → operation → output graphs, static validation, revision-2 save/reload. |
| NW-T3-12 | L5 runtime/profile | PASS | 14 browser cases plus 21 native TensorFlow/PyTorch/JAX CPU cases; two browser profiles only. |
| NW-T3-12 | L6 persistence/recovery | PASS | Exact graph receipts, diagnostics, project deletion, process shutdown, and port release. |

The new authoritative browser spec is [`mathematical-unary-families-slice.spec.ts`](../../app/client/e2e/mathematical-unary-families-slice.spec.ts). The graph-aware Functional Validation fixture uses deterministic ones only for `math.log`, `math.log1p`, `math.sqrt`, `math.rsqrt`, and `math.reciprocal`; unrelated graphs retain the historical zero fixture. Focused backend tests cover catalog metadata, implementation mappings, emitted symbols, static shape/dtype, and native forward execution across TensorFlow, PyTorch, and JAX. No catalog, emitter, shape-inference, or Builder defect was found.

Before the comparison/reduction continuation, campaign arithmetic was corrected:
46 named strategy slices plus the separate recurrent-wrapper continuation make
47 actual slices, with 29 PASS, 4 PARTIAL, 0 FAIL, 0 BLOCKED, and 14 UNTESTED.
The pre-run 49-slice aggregate and 48 displayed status sum were bookkeeping
errors; the continuation and its final aggregate are recorded below.

## Current bounded run: NW-T3-13/NW-T3-14 comparison, logical, and reduction families

Date: 2026-09-29
Repository: `develop`; validation checkout before scoped changes: `264fbc81f55bfd77c44dab4b9ee9227f66b45ecf`
Run report: [NW-T3-13/NW-T3-14 comparison and reduction validation](comprehensive-validation-20260929/NW-T3-13-14.md)
Run manifest: [NW-T3-13/NW-T3-14 run manifest](comprehensive-validation-20260929/NW-T3-13-14-run-manifest.json)
Catalog: 149 nodes; 73 Mathematical entries; normalized fingerprint `acf538fb04f50552e1577299aa3f0cc09511d4e7d0937f73aa8418125a3a9ef8`
Profiles: Keras API / PyTorch backend / CPU and native PyTorch / CPU browser lanes; native TensorFlow/PyTorch/JAX CPU operation matrix
Evidence root: `assets/QA/comprehensive-validation-20260929/evidence/nw-t3-13-14-comparison-reduction/`

| Slice | Level | Result | Bounded evidence |
| --- | --- | --- | --- |
| NW-T3-13 | L0 catalog inventory | PASS | Nine live comparison/logical definitions, verified mappings, static contracts, rendered Mathematical Builder category, and live catalog fingerprint. |
| NW-T3-13 | L1 node behavior | PASS | Nine elementwise nodes materialized and runtime-validated on the two exact browser lanes; boolean truth tables pass. |
| NW-T3-13 | L2 parameter behavior | PASS | Parameter-free comparison/logical contracts and float/bool input dtype boundaries; broader dtype/rank state coverage remains open. |
| NW-T3-13 | L3 ports/edges | PASS | Ordered binary tensor-list inputs and output terminal connected and statically validated for every selected definition. |
| NW-T3-13 | L4 composition | PASS | Input → operation → output graphs, revision-2 save/reload, exact static and runtime shapes. |
| NW-T3-13 | L5 runtime/profile | PASS | 18 positive browser cases plus 27 native TensorFlow/PyTorch/JAX CPU cases; browser coverage is limited to Keras/PyTorch and native PyTorch. |
| NW-T3-13 | L6 persistence/recovery | PASS | Functional Validation, exact receipts, empty diagnostics, project deletion, process shutdown, and port release. |
| NW-T3-14 | L0 catalog inventory | PASS | Twelve live reduction/aggregate definitions, verified mappings, static contracts, rendered Mathematical Builder category, and live catalog fingerprint. |
| NW-T3-14 | L1 node behavior | PASS | Twelve reduction nodes materialized and runtime-validated on the two exact browser lanes; numeric and boolean aggregate values pass. |
| NW-T3-14 | L2 parameter behavior | PASS | Selected axis/keepdims cases plus generic `math.reduce` sum; broader axes, ranks, dtypes, and parameter combinations remain open. |
| NW-T3-14 | L3 ports/edges | PASS | Single tensor input and output terminal connected and statically validated for every selected definition. |
| NW-T3-14 | L4 composition | PASS | Input → reduction → output graphs, revision-2 save/reload, reduced static and runtime shapes. |
| NW-T3-14 | L5 runtime/profile | PASS | 24 positive browser cases plus 36 native TensorFlow/PyTorch/JAX CPU cases; browser coverage is limited to Keras/PyTorch and native PyTorch. |
| NW-T3-14 | L6 persistence/recovery | PASS | Functional Validation, exact receipts, empty diagnostics, project deletion, process shutdown, and port release. |

The authoritative serial browser run passed 44/44: one live catalog/render
case, 42 positive exact-profile cases, and one cleanup case. The backend
operation file passed 260 tests; frontend unit, lint, and build gates also
passed. The first browser attempt exposed and corrected a test-harness catalog
locator mismatch; the repaired focused run and final full run passed. No
product defect was found in the selected catalog, emitter, shape, Builder,
persistence, or runtime paths.

At the pre-2026-09-30 checkpoint, the campaign inventory was 47 actual slices:
31 PASS, 4 PARTIAL, 0 FAIL, 0 BLOCKED, and 12 UNTESTED. The continuation
below supersedes that checkpoint for the Tier 3 scope; the historical bounded
run text is retained for provenance.

## Current run: NW-T3-15 through NW-T3-17 mathematical operations

Date: 2026-09-30
Repository: `develop`; source baseline `416c158c61349d7a1106e71e64fba4174d3178a5` before scoped campaign changes
Run report: [NW-T3-15 through NW-T3-17](comprehensive-validation-20260930/NW-T3-15-17-mathematical.md)
Catalog crosswalk: [full 149-definition catalog crosswalk](comprehensive-validation-20260930/evidence/full-tier3-catalog-crosswalk.json); the Mathematical slice crosswalk remains [catalog-crosswalk.json](comprehensive-validation-20260930/evidence/catalog-crosswalk.json)
Catalog: 149 nodes; 73 Mathematical entries; normalized fingerprint `acf538fb04f50552e1577299aa3f0cc09511d4e7d0937f73aa8418125a3a9ef8`
Profiles: Keras API / PyTorch backend / CPU and native PyTorch / CPU browser lanes
Evidence root: `assets/QA/comprehensive-validation-20260930/evidence/nw-t3-15-17-mathematical/`

| Slice | Level | Result | Bounded evidence |
| --- | --- | --- | --- |
| NW-T3-15 | L0-L6 | PASS | Six live arithmetic/dot/matrix definitions; 12 exact serial Builder/runtime cases; parameters, shapes, invalid dimensions, persistence, and cleanup. |
| NW-T3-16 | L0-L6 | PASS | Eight live cast/clip/concatenate/dimension/reshape/repetition/stack/transpose/tile definitions; 16 exact serial cases with axis, rank, dtype, repeat/tile, persistence, and cleanup evidence. |
| NW-T3-17 | L0-L6 | PASS | Two remaining live Mathematical definitions; four exact serial cases covering `take`, `stop_gradient`, static/runtime shapes, persistence, and safe-expression carry-forward. |

The final aggregate passed 32/32 serial cases. The crosswalk reports no missing
or duplicate Mathematical definition IDs. Full six-profile/export, worker,
accessibility, and Tier 4 capability coverage remain separate boundaries.

## Current run: NW-T3-18 and NW-T3-19 output and graph integrity

Date: 2026-09-30
Repository: `develop`; source baseline `416c158c61349d7a1106e71e64fba4174d3178a5` before scoped campaign changes
Run report: [NW-T3-18/NW-T3-19 output and graph integrity](comprehensive-validation-20260930/NW-T3-18-19-output-graph-integrity.md)
Generated edge campaign: [nw-t3-19-generated-edge-campaign.json](comprehensive-validation-20260930/evidence/nw-t3-19-generated-edge-campaign.json)

| Slice | Level | Result | Bounded evidence |
| --- | --- | --- | --- |
| NW-T3-18 | L0-L6 | PASS | Inferred terminal outputs, multiple explicit outputs, recurrent state outputs, incompatible task/output validation, persistence, repair, and explicit-output browser transitions. |
| NW-T3-19 | L0-L6 | PASS | Exhaustive live port inventory; generated contract-class candidates; compatible edges, fan-in/fan-out, rank/dimension/dtype/cardinality negatives, cycles, dangling endpoints, unreachable components, deleted-node revalidation, and profile restrictions. |

The generated inventory covers 149 definitions, 146 input ports, 155 output
ports, and 22,630 output/input candidates. Dynamic shape, dtype, graph-state,
and profile negatives are partitioned into focused backend/browser cases rather
than falsely represented as a literal Cartesian execution.

## Tier 3 reconciliation

Date: 2026-09-30

The full live catalog crosswalk contains all 149 definitions exactly once;
its reconciliation reports no missing, duplicate, or stale-unclassified IDs.
The Mathematical category contains all 73 definitions exactly once, with no
missing, duplicate, or stale-unclassified IDs. The recurrent-wrapper continuation is PASS under
the corrected standard-LSTM acceptance criterion. NW-T3-15 through NW-T3-19
are PASS, and the formal Tier 3 set plus the separately tracked wrapper
continuation is **20 PASS, 0 PARTIAL, 0 UNTESTED** within this campaign's
documented profile and evidence boundaries.

The campaign intentionally does not promote NW-T4-01/T4-02, PyTorch Inductor,
export-matrix closure, worker concurrency, or accessibility into Tier 3.

## Future session run template

Copy this block for each validation session. Do not overwrite prior run blocks.

### Run ID: YYYY-MM-DD-short-name

| Field | Value |
| --- | --- |
| Date/time |  |
| Branch/commit |  |
| Working tree |  |
| Launcher and ports |  |
| Catalog count/fingerprint |  |
| Runtime/package versions |  |
| Device/compiler state |  |
| Database/test isolation |  |
| Profiles executed |  |
| QA artifact directory |  |

#### Run result

| Area | Result | Evidence |
| --- | --- | --- |
| L0 catalog inventory | PENDING |  |
| L1 every node | PENDING |  |
| L2 parameters | PENDING |  |
| L3 port/edge combinations | PENDING |  |
| L4 graph families | PENDING |  |
| L5 six profiles | PENDING |  |
| L6 persistence/recovery | PENDING |  |
| Open failures/blockers |  |  |

#### Closure note

State exactly which rows are PASS, EXPECTED_UNSUPPORTED, BLOCKED, FAIL, STALE, or still PENDING. If a run is partial, say so explicitly and carry unresolved rows forward.

## Future session checklist

- Inspect project_status_ledger.md and this ledger before starting.
- Freeze the live catalog and profile capability snapshot.
- Generate/update the per-node and per-edge sheets from that snapshot.
- Run static contract checks before browser work.
- Use the official launcher and the real Builder workflow for catalog, connections, save, validation, and profile switching.
- Run each supported profile through source and Functional Validation; keep compiler/export availability separate.
- Record expected unsupported and blocked paths instead of treating them as failures or successes.
- Save all evidence under assets/QA and update the canonical project status only after reviewing the complete run.
- Stop task-owned services and helpers, then record cleanup.
