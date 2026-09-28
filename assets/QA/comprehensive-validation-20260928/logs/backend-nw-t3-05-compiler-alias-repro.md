# NW-T3-05 compiler-alias reproduction

Command:

```text
app/server/.venv/Scripts/python.exe -m pytest app/tests/test_model_builder_export_regressions.py::test_native_tensorflow_onnx_export_inlines_jit_call_graph -q
```

Before the controller fix, the test reproduced consistently with:

```text
RuntimeControllerError: TensorFlow tf.function / XLA is not operational for tensorflow
```

The live probe was available, but worker metadata used `tensorflow_jit` while
the controller looked up only the UI compiler ID `tensorflow_function`. The
same mismatch existed for `torch_compile` versus `pytorch_compile`.

After changing the controller to use the existing alias-aware capability
resolver, the exact export test passed. The repository-wide backend suite also
completed with exit code 0. The final focused results are in
[backend-nw-t3-05-focused-after-alias-fix.log](backend-nw-t3-05-focused-after-alias-fix.log).
