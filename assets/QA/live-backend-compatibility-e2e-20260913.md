# Live backend compatibility E2E report

Date: 2026-09-13

Scope: current develop checkout at 1331a42 (Document backend compatibility consolidation), including the seven local commits from 269880f through 1331a42.

The application was launched with the repository launcher on 127.0.0.1:9750 / 127.0.0.1:6425 and piloted through Chrome using the in-app computer-use/browser tooling. No application source was modified. The test models created for this run remain in the local model catalog so the UI evidence is inspectable:

- QA Backend Matrix 20260913, id 01d4c22c-1af7-4f69-9b6c-fa416ad4bd14
- QA Sequence Compatibility 20260913, id 65e99d63-3a90-4e53-8375-aaa857db6f7f
- QA LSTM Parameter Matrix 20260913, id 4e84e851-493c-44b2-af5a-ebfee3c27d11

## Runtime and compatibility model

Live Environment UI evidence reported Keras and the native framework API operational, with all three frameworks usable:

- TensorFlow 2.21.0, CPU
- PyTorch 2.13.0+cu126, CPU and NVIDIA GeForce RTX 3060 Laptop GPU
- JAX 0.11.0, cpu:0

The implementation defines six execution profiles:

| Profile | Backend | Keras | Catalog result | E2E result |
|---|---|---:|---:|---|
| Keras/TensorFlow | tensorflow | on | 139 available / 6 unavailable | PASS |
| Keras/PyTorch | pytorch | on | 139 available / 6 unavailable | PASS |
| Keras/JAX | jax | on | 139 available / 6 unavailable | PASS |
| Native TensorFlow | tensorflow | off | 107 available / 38 unavailable | PASS |
| Native PyTorch | pytorch | off | 107 available / 38 unavailable | PASS |
| Native JAX | jax | off | 107 available / 38 unavailable | PASS |
| Keras absent/unavailable | any | unavailable | Not emulatable through the supported UI; live discovery reported Keras operational | NOT TESTED |

There were no implementation-level availability differences among native TensorFlow, native PyTorch, and native JAX. Their differences are compiler settings, parameter capabilities, and export formats.

## Complete UI-created workflow

The primary model was created entirely in the Builder:

input.tabular(shape=[8]) -> layer.dense(units=2, activation=softmax, use_bias=true) -> output.classification

It retained 3 nodes, 2 connections, inferred shapes [?, 8] and [?, 2], and 18 parameters through profile changes.

| Profile | Validation | Source compilation | Smoke execution | Backend evidence |
|---|---|---|---|---|
| Keras/TensorFlow | 0 diagnostics | PASS; generated source set KERAS_BACKEND='tensorflow' | PASS; CPU, 1 input, 1 output, 18 parameters, 383.1 ms | Keras TensorFlow worker |
| Keras/PyTorch | 0 diagnostics | PASS; generated source set KERAS_BACKEND='torch'; Adam learning rate 0.0005 and AUC retained | PASS; CPU, 1 input, 1 output, 18 parameters, 2408.1 ms | Keras PyTorch worker |
| Keras/JAX | 0 diagnostics | PASS; generated source set KERAS_BACKEND='jax' | PASS; cpu:0, 1 input, 1 output, 18 parameters, 2550.5 ms | Keras JAX worker |
| Native TensorFlow | 0 diagnostics | PASS; generated native TensorFlow source with tf.function(jit_compile=True) | PASS; CPU, 1 input, 1 output, 18 parameters, 968.6 ms | Native TensorFlow worker |
| Native PyTorch | 0 diagnostics | PASS; generated native PyTorch module and optimizer source | PASS; CPU, 1 input, 1 output, 18 parameters, 1073.7 ms | Native PyTorch worker |
| Native JAX | 0 diagnostics | PASS; generated Flax source with jax.jit(model.apply) | PASS; cpu:0, 1 input, 1 output, 18 parameters, 1069.7 ms | Native JAX worker |

The backend worker job list corroborated the UI results: all six smoke jobs completed with state=succeeded, exit code 0, expected backend/API, and matching input/output metadata.

## Backend-specific node and parameter matrix

| Feature | Intended restriction | Actual UI behavior | Validation/compile behavior | Status |
|---|---|---|---|---|
| recurrent.simple_rnn | Keras profiles only; no native mapping | Keras catalog item active. After switching to each native profile, the retained node showed UNAVAILABLE, all inspector controls were disabled, and the catalog item was Limited/disabled with the selected backend in its title | Exact diagnostic, for example Simple RNN is not supported by native pytorch.; saved invalid revisions showed Compile source disabled | PASS |
| conv.average_pool1d | Keras mapping only; native source mapping absent | Nodes library showed Keras Supported and native TensorFlow/PyTorch/JAX Unavailable. Builder showed the item active with Keras and Limited/disabled under native PyTorch | Generic profile resolver prevents adding it to native profiles | PASS for catalog restriction |
| recurrent.lstm | Native mapping supports only canonical options | UI-created LSTM sequence graph compiled and smoked with Keras/PyTorch. Native PyTorch kept unit_forget_bias=false editable and valid; native TensorFlow/JAX disabled that parameter | Native PyTorch source compiled. TensorFlow/JAX produced Unit forget bias has no supported native ... mapping for this value and blocked the profile | PASS |
| Dense kernel_initializer | Keras supports non-default initializers; native inference owns deterministic parameters | he_normal was editable and retained under Keras; source emitted kernel_initializer='he_normal'. After native switching, the value remained visible but the control was disabled | Native PyTorch blocked with Parameter 'Kernel initializer' is not supported for native pytorch; restoring glorot_uniform made the same graph valid and compilable | PASS |
| Native recurrent options | Canonical native parity only | return_state, go_backwards, stateful, unroll, seed, recurrent dropout and unsupported initializers/regularizers/constraints were disabled in the LSTM inspector | Invalid non-default values are retained for repair and block validation | PASS for exercised LSTM path |

The full catalog inventory derived from the live catalog contains six Keras-unavailable definitions:

input.constant, input.parameter, math.index, merge.average, merge.split, output.detection_placeholder.

All native profiles share these additional unavailable definitions:

conv.average_pool1d, conv.average_pool2d, conv.average_pool3d, conv.conv3d, conv.cropping, conv.depthwise, conv.global_pool1d, conv.global_pool2d, conv.global_pool3d, conv.max_pool1d, conv.max_pool3d, conv.padding, conv.separable, conv.transposed, conv.upsampling, layer.group_norm, layer.permute, layer.repeat, layer.reshape, recurrent.bidirectional, recurrent.simple_rnn, recurrent.stack, transformer.attention_mask, transformer.decoder_block, transformer.encoder_block, transformer.feed_forward, transformer.positional_encoding, transformer.residual, transformer.stack, attention.cross, attention.multi_head, and attention.self.

Other declared native parameter restrictions were verified from the live catalog and the shared inspector resolver: native PyTorch Dense/Conv1D/Conv2D activations are limited to linear, relu, gelu, sigmoid, tanh, and softmax; native BatchNorm restricts axis/affine/synchronization options; native Embedding does not propagate masks or LoRA; native Flatten uses channel-last ordering for TensorFlow/JAX; native LayerNorm uses the trailing axis; and native GRU/LSTM use canonical activations, zero recurrent dropout, and canonical reset/reset-forget options. The UI-created LSTM and Dense cases exercised the most consequential parameter paths; architecture-only definitions were tested through the live catalog/library restriction rather than compiled individually because the implementation intentionally marks them unavailable.

## Invalid, mixed, and recovery workflows

- Empty graph: validation returned two graph-level errors for missing input and missing output/terminal binding.
- Unconnected graph: validation returned required-input, unreachable-output, and input-not-contributing diagnostics.
- Invalid parameter: Units=0 showed Enter a complete value within the supported range. and disabled the Validate button.
- Classification mismatch: changing the final Dense activation to relu returned both output.classification requires a final softmax activation and Loss 'categorical_crossentropy' requires one of: softmax; source compilation was disabled.
- Unsupported node: native Simple RNN was retained rather than silently removed, marked unavailable, and prevented compilation.
- Unsupported parameter: native PyTorch retained he_normal, disabled the control, and blocked the profile until restored to the native default.
- Backend switching: Keras/PyTorch, Keras/TensorFlow, Keras/JAX, and all three native profiles were exercised. Common nodes, connections, labels, shapes, and values survived; validation and generated source followed the newly selected profile.
- Recovery: correcting activation, restoring the initializer, or re-enabling Keras made the same graph valid without recreation.

## Compilation and export

Native compiler settings were exercised through the UI:

- TensorFlow XLA source was generated and native TensorFlow smoke succeeded.
- JAX JIT source was generated and native JAX smoke succeeded.
- PyTorch native source was generated with AdamW and native PyTorch smoke succeeded.
- Keras/PyTorch ONNX interchange export completed and the UI reported a verified download.
- Native PyTorch ExportedProgram export completed and the UI reported a verified .pt2 download.
- Native TensorFlow SavedModel export completed and the UI reported a verified .savedmodel.zip download.
- Native JAX Flax checkpoint export completed and the UI reported a verified .flax-checkpoint.zip download.
- Export capability gating matched the selected profile: Keras, TensorFlow, PyTorch, and Flax formats were enabled or disabled with profile-specific reasons.

The non-operational PyTorch torch.compile option was deliberately tested. The UI displayed the warning, allowed the setting to be saved and source-generated for later runtime use, and the worker then rejected the smoke request with the explicit user-visible message PyTorch torch.compile is not operational for pytorch. Disabling it restored successful compilation and smoke execution. This is an intentional deferred-runtime boundary in the current implementation, not a silent success.

## Regression suite and defects

Browser console logs were empty for the live manual run. The latest 1,000 backend log lines had no errors for the manual workflows; the deliberate torch.compile request returned an expected HTTP 422 without a traceback. An older unrelated graph replacement failed record referenced a different project id and was not reproducible for these test models.

The current Playwright workflow file was also exercised:

- Default configuration (fullyParallel=true, six workers): 12 passed, 2 failed in 40 seconds.
- The two failures were deletes a model from its catalog card without navigating and renders stable connection previews, fan-out connections, and cycle rejection feedback.
- Each failed test passed in isolation with one worker.
- Full workflows.spec.ts run with --workers=1: 14 passed in 1.4 minutes.

### QA-1 — shared SQLite parallel-fixture lock

Severity: S3, non-blocking test-harness/reliability issue.

Reproduction: run npm exec -- playwright test e2e/workflows.spec.ts --reporter=line with the default workers.

Expected: parallel fixture creation/deletion should not return a 500.

Actual: the catalog delete test received 500 from DELETE /api/projects/{id} with sqlite3.OperationalError: database is locked; the same id returned 204 immediately afterward, and the test passes serialized.

Likely cause: app/client/playwright.config.ts enables parallel workers while all fixtures share the single backend/SQLite database at port 9750. app/server/repositories/backend.py configures a 30-second SQLite/busy timeout but does not serialize concurrent write transactions or isolate test databases.

Recommended fix: serialize destructive fixture operations, use an isolated database per worker, or add an application-level SQLite write lock/retry policy appropriate for the local runtime.

### QA-2 — connector readiness assertion is timing-sensitive under parallel load

Severity: S3, non-blocking E2E reliability issue.

Reproduction: the same default parallel Playwright run can fail at workflows.spec.ts:358 because the target connector lacks the f-connector-connectable class immediately after drag; the DOM contained only f-component f-connector f-connector-target. The test passes with one worker.

Likely cause: graph-canvas.component.ts asynchronously refreshes connection candidates and binds them through fCanBeConnectedTo; concurrent shared-backend activity increases the window in which Foblex renders the target before the candidate map is applied. The test currently waits for canvas fit but not for candidate-resolution readiness.

Recommended fix: expose/await a deterministic connection-candidate-ready state before the drag assertion, or make the test wait for the authoritative target class/candidate response.

### Known deferred boundary — unavailable torch.compile

The configuration dialog intentionally says settings can be saved for a later runtime when the compiler is unavailable. compilation_capabilities.py marks the capability unavailable and runtime_controller.py rejects execution with native_compiler_unavailable. The observed behavior is explicit and recoverable, but an earlier validation error would improve user feedback.

## Release verdict

READY WITH NON-BLOCKING ISSUES

All six supported backend/Keras execution profiles passed real UI creation, validation, source compilation, worker execution, profile switching, and relevant export checks. Invalid graphs and incompatible nodes/parameters were blocked and repaired correctly. The remaining issues are parallel SQLite/connector-test reliability and the intentionally deferred, clearly surfaced PyTorch torch.compile capability.
