# Model Builder validation ledger

Last updated: 2026-09-29

Preparation status: ACTIVE. The bounded current-checkout runs (`NW-T3-01` through `NW-T3-06`) are recorded below; full closure remains pending.

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
| GF-009 | Two branches → merge/concatenate/stack/residual/skip → output | fan-in, shape equality, axis, tensor-list cardinality | Backend compatibility evidence; current NW-T3-01 Concatenate case | PASS for the current K/PyTorch tensor-list Concatenate case; expand merge/axis/cardinality matrix |
| GF-010 | Input → mathematical unary/binary/reduction/tensor operations | dtype/rank/shape rules and safe expressions | Native parity supporting evidence; current NW-T3-01 Concatenate case | PASS for the current K/PyTorch Concatenate path only; full mathematical family remains pending |
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

## Current bounded run: recurrent wrapper continuation

Date: 2026-09-28
Repository: `develop` at implementation commit `38c9ccc`
Run report: [bounded recurrent wrapper continuation](comprehensive-validation-20260928/NW-T3-recurrent-wrappers.md)
Run manifest: [recurrent wrapper manifest](comprehensive-validation-20260928/NW-T3-recurrent-wrappers-run-manifest.json)
Worker matrix: [NW-T5-02 disposable worker matrix](comprehensive-validation-20260928/NW-T5-02-workers.md)
Catalog: 149 nodes; fingerprint `e0ea2f09ef44a4b3fcb10988495fcdea578f6d504c9bc39915e224d50276f439`
Profile: `execution_backend=pytorch`, `keras_enabled=true`, Keras API / PyTorch backend, CPU worker
Evidence root: `assets/QA/comprehensive-validation-20260928/`

| Level | Bounded result | Closure boundary |
| --- | --- | --- |
| L0 catalog inventory | PASS | The live catalog count and fingerprint match the freeze; node-level coverage remains open. |
| L1 single-node behavior | PASS for Bidirectional Recurrent Layer, Recurrent Stack, and retained Simple RNN/GRU/LSTM/Concatenate neighbors | The remaining definitions and profile-specific enable/disable behavior remain PENDING. |
| L2 parameter behavior | PARTIAL | All bidirectional merge shapes, standard LSTM counts, stack layers, and invalid-layer diagnostics pass. The plan's requested LSTM counts `64`/`68` conflict with standard Keras LSTM semantics; observed static/runtime counts are `256`/`272`. |
| L3 port/edge behavior | PASS for the selected sequence input/wrapper/output graphs | Exhaustive catalog edge generation and incompatible-pair coverage remain PENDING. |
| L4 graph composition | PASS for the selected bidirectional and two-layer stack graphs | Full graph-family and profile composition remain PENDING. |
| L5 runtime/profile | PASS for the selected Keras/PyTorch CPU lane | Source/export, profile switching, the other five profiles, and positive PyTorch Inductor remain PENDING/PARTIAL at their existing boundaries. |
| L6 persistence/recovery | PASS for blocked stack repair, save/reopen at revision 2, validation, runtime, and fixture cleanup | Matrix-wide recovery and failure cleanup remain PENDING. |

The serial authoritative catalog run passed `6/6` cases and captured rendered
Builder/runtime screenshots plus empty browser-error payloads. The focused
backend/export/native-parity suite, changed-file Ruff, changed-spec ESLint,
frontend unit/lint/build gates passed. Full strict Pyright remains PARTIAL at
58 existing errors. The separate unchanged `workflows.spec.ts` worker matrix
was not clean: `19/20`, `18/20`, and `17/20` passed at workers 1, 2, and 4;
no SQLite lock signatures were found, but active-job and parallel Builder
failures keep `NW-T5-02`/`ISSUE-001` open. The bounded basic-convolution
`NW-T3-03` slice was subsequently selected and passed; the advanced/spatial
convolution `NW-T3-04` slice has now also passed, and the bounded recurrent
`NW-T3-05` slice is recorded below. The bounded `NW-T3-06` attention primitive
slice is now recorded below.

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

The authoritative serial browser matrix passed 11/11 cases. The cases covered Max Pooling 1D, Average Pooling 2D, Max Pooling 3D, Depthwise, Separable, and Transposed Convolution, Upsampling, Padding, Cropping, and a native-PyTorch Depthwise neighbor. Every case produced the expected static shape and parameter count, saved with three nodes and two edges, reloaded with three nodes, completed with the exact selected worker, and emitted an empty browser error payload. The in-app Browser observation confirmed the rendered live convolutional catalog. Full Ruff is green after three narrow import-block repairs; strict Pyright remains PARTIAL at 58 existing errors. The existing recurrent-wrapper, torch.compile, and 1/2/4-worker matrix limitations remain unchanged. The next selected catalog slice was NW-T3-06 attention primitives; that bounded run is recorded below. This remains a bounded PASS, not full Model Builder certification.

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
