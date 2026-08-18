# Native target audit — 2026-08-18

Commit under test: `615cd03` (`develop`, equal to `origin/develop` before this checkpoint).

This audit covers the native TensorFlow and JAX/Flax exporters, their effective manifest capabilities, and the dependency gate used by Keras runtime workers. It does not promote a target based on generated-source inspection.

## Dependency and worker evidence

Command, run from `app/server` with the repository environment:

```powershell
\.venv\Scripts\python.exe -c "import importlib, importlib.metadata as md; ..."
```

Observed package state:

| Package | Result |
| --- | --- |
| `keras` | importable, `3.15.1` |
| `torch` | importable, `2.13.0+cpu` |
| `tensorflow` | not importable: `ModuleNotFoundError: No module named 'tensorflow'` |
| `jax` | not importable: `ModuleNotFoundError: No module named 'jax'` |
| `flax` | not importable: `ModuleNotFoundError: No module named 'flax'` |

Live `KerasBackendRunner().probe_statuses()` result:

```json
{
  "tensorflow": {"installed": false, "available": false, "reason_code": "dependency_missing"},
  "pytorch": {"installed": true, "available": true, "version": "2.13.0", "accelerator": "cpu", "device_name": "CPU"},
  "jax": {"installed": false, "available": false, "reason_code": "dependency_missing"}
}
```

The exact blocker for native TensorFlow is the missing `tensorflow` package. The exact blocker for native JAX/Flax is the missing `jax` and `flax` packages. No corresponding generated source was executed because the required runtime cannot be imported in this environment.

## Effective manifest inventory

`get_node_catalog()` loaded 85 manifests. Effective node support is:

| Target | Supported nodes | Disabled nodes |
| --- | ---: | ---: |
| Keras | 31 | 54 |
| Native PyTorch | 27 | 58 |
| Native TensorFlow | 0 | 85 |
| Native JAX / Flax | 0 | 85 |

Keras has no explicit disabled parameter capabilities. TensorFlow and JAX have no enabled node or parameter capabilities. The disabled node IDs for Keras and PyTorch are listed below; TensorFlow and JAX disable every one of the 85 manifest IDs.

### Keras-disabled node IDs

`conv.average_pool1d`, `conv.average_pool2d`, `conv.average_pool3d`, `conv.conv1d`, `conv.conv3d`, `conv.cropping`, `conv.depthwise`, `conv.global_pool1d`, `conv.global_pool2d`, `conv.global_pool3d`, `conv.max_pool1d`, `conv.max_pool3d`, `conv.padding`, `conv.separable`, `conv.transposed`, `conv.upsampling`, `layer.group_norm`, `layer.permute`, `layer.repeat`, `layer.reshape`, `input.constant`, `input.parameter`, `math.abs`, `math.argmax`, `math.exp`, `math.log`, `math.power`, `math.reduce`, `math.expression`, `math.softmax`, `math.sqrt`, `math.index`, `math.slice`, `math.transpose`, `merge.average`, `merge.divide`, `merge.dot`, `merge.matmul`, `merge.maximum`, `merge.minimum`, `merge.multiply`, `merge.split`, `merge.stack`, `merge.subtract`, `output.detection_placeholder`, `recurrent.bidirectional`, `recurrent.stack`, `recurrent.simple_rnn`, `transformer.attention_mask`, `transformer.stack`, `attention.cross`, `transformer.feed_forward`, `transformer.residual`, `transformer.decoder_block`.

### Native-PyTorch-disabled node IDs

The Keras-disabled list above, plus `attention.multi_head`, `transformer.positional_encoding`, `attention.self`, and `transformer.encoder_block`.

### Native-TensorFlow and native-JAX-disabled node IDs

All 85 manifest IDs are disabled. This includes every input, output, core, convolutional, recurrent, attention, transformer, merge, and mathematical node; the zero-support state is intentional until dependency-backed runtime tests pass.

## Explicitly disabled PyTorch parameters

These parameter IDs are disabled in the effective PyTorch mappings; their manifest values are rejected when non-default rather than silently ignored:

| Node | Disabled parameter IDs |
| --- | --- |
| `conv.conv2d` | `kernel_initializer`, `bias_initializer`, `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`, `kernel_constraint`, `bias_constraint` |
| `layer.batch_norm` | `beta_initializer`, `gamma_initializer`, `moving_mean_initializer`, `moving_variance_initializer`, `beta_regularizer`, `gamma_regularizer`, `beta_constraint`, `gamma_constraint` |
| `layer.dense` | `kernel_initializer`, `bias_initializer`, `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`, `kernel_constraint`, `bias_constraint` |
| `layer.dropout` | `seed` |
| `layer.embedding` | `embeddings_initializer`, `embeddings_regularizer`, `activity_regularizer`, `embeddings_constraint`, `lora_rank` |
| `layer.layer_norm` | `beta_initializer`, `gamma_initializer`, `beta_regularizer`, `gamma_regularizer`, `beta_constraint`, `gamma_constraint` |
| `recurrent.gru` | `kernel_initializer`, `recurrent_initializer`, `bias_initializer` |
| `recurrent.lstm` | `kernel_initializer`, `recurrent_initializer`, `bias_initializer` |

The enabled-but-restricted PyTorch parameters are also target-validated: BatchNorm `axis` is `-1` or `1`, center/scale must be `true`, and `synchronized` must be `false`; Embedding `mask_zero` must be `false`; LayerNorm uses trailing-axis `axis=[-1]` with center/scale `true`; recurrent activation/recurrent-activation are `tanh`/`sigmoid`, dropout values are `0.0`, `return_state`, `go_backwards`, `stateful`, and `unroll` are `false`, recurrent seeds are `null`, and the GRU `reset_after` value is `true`.

## Static exporter findings

The native TensorFlow exporter contains handlers for Dense, Dropout, Flatten, Embedding, Conv2D, MaxPool2D, normalization, recurrent, and common merge nodes, but its generated code does not yet preserve the full canonical parameter contract (for example Dense bias, Dropout training semantics, and several convolution/pooling/layout settings). The JAX exporter similarly fixes or omits behavior such as Dense bias, deterministic Dropout, running-average BatchNorm, pooling strides/padding, convolution options, and recurrent sequence semantics. These findings are why the exporters remain unadvertised even though source strings can be produced by direct class invocation.

## Regression gate

`app/tests/test_v2_contract.py::test_unproven_native_targets_remain_disabled` asserts that every manifest keeps TensorFlow and JAX disabled. Re-enable a node only after its generated source executes with the repository-declared optional dependency and its representative compositions and parameter/error cases pass.
