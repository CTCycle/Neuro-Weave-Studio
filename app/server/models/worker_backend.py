# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOperatorIssue=false

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from common.catalogs.loader import get_node_catalog
from domain.compilation_catalog import CompilationCatalog
from domain.enums import ExecutionBackend, keras_backend_value
from domain.exports import ExportFormat, ExportOptions
from domain.graph import ModelGraph
from domain.runtime import MemoryPolicyKind, RuntimeApi, WorkerConfiguration
from models.tensor_inspection_runtime import persist_inspection
from validation.output_resolution import input_nodes, output_selectors, resolved_outputs

###############################################################################
class BackendWorker(Protocol):
    """Common contract implemented by one isolated execution-backend worker."""

    backend: ExecutionBackend

    # -------------------------------------------------------------------------
    def configure(self, config: WorkerConfiguration) -> None: ...

    # -------------------------------------------------------------------------
    def capabilities(self) -> dict[str, object]: ...

    # -------------------------------------------------------------------------
    def native_compilation_capabilities(self) -> dict[str, bool]: ...

    # -------------------------------------------------------------------------
    def device_inventory(self) -> list[dict[str, object]]: ...

    # -------------------------------------------------------------------------
    def memory_metrics(self) -> dict[str, int | None]: ...

    # -------------------------------------------------------------------------
    def device_scope(self) -> AbstractContextManager[Any]: ...

    # -------------------------------------------------------------------------
    def synchronize(self, values: list[Any]) -> None: ...

    # -------------------------------------------------------------------------
    def run_smoke(
        self,
        graph: ModelGraph,
        checkpoint: Callable[[], None],
        *,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> dict[str, object]: ...

    # -------------------------------------------------------------------------
    def run_inspection(
        self,
        graph: ModelGraph,
        checkpoint: Callable[[], None],
        *,
        seed: int,
        project_id: str,
        project_revision: int,
        output_root: Path,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> dict[str, object]: ...

    # -------------------------------------------------------------------------
    def run_export(
        self,
        graph: ModelGraph,
        destination: Path,
        export_format: ExportFormat,
        options: ExportOptions,
        checkpoint: Callable[[], None],
        project_id: str | None = None,
        project_revision: int | None = None,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> dict[str, object]: ...

    # -------------------------------------------------------------------------
    def cleanup(self) -> None: ...


_BACKEND_MODULES = {
    ExecutionBackend.TENSORFLOW: "models.tensorflow_worker",
    ExecutionBackend.PYTORCH: "models.pytorch_worker",
    ExecutionBackend.JAX: "models.jax_worker",
}

###############################################################################
def configure_startup_environment(config: WorkerConfiguration) -> None:
    """Set process-wide visibility before the selected backend module is loaded."""

    if config.api is RuntimeApi.KERAS:
        os.environ["KERAS_BACKEND"] = keras_backend_value(config.backend)
    else:
        # A native worker must not inherit a Keras backend selection from the
        # server process. Native modules import only their framework API.
        os.environ.pop("KERAS_BACKEND", None)

    if config.device_id == "auto":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif config.device_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if config.backend is ExecutionBackend.JAX:
            os.environ["JAX_PLATFORMS"] = "cpu"
    elif config.device_id.startswith(("cuda:", "rocm:")):
        os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id.split(":", 1)[1]
    elif config.device_id.startswith(("mps:", "xpu:")):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        raise RuntimeError(f"Unsupported worker device '{config.device_id}'")

    policy = config.memory_policy.kind
    if policy is MemoryPolicyKind.JAX_NO_PREALLOCATION:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    elif policy is MemoryPolicyKind.JAX_PREALLOCATE_FRACTION:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.memory_policy.fraction)
    elif policy is MemoryPolicyKind.JAX_PLATFORM_ALLOCATOR:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

###############################################################################
def load_backend_worker(backend: ExecutionBackend) -> BackendWorker:
    """Load exactly one optional framework implementation in the fresh worker."""

    module = importlib.import_module(_BACKEND_MODULES[backend])
    factory = cast(Callable[[], BackendWorker], module.__dict__["create_worker"])
    return factory()

###############################################################################
def run_keras_smoke(
    graph: ModelGraph,
    *,
    keras_module: Any,
    build_model: Callable[..., Any],
    compile_model: Callable[..., Any],
    capabilities: dict[str, object],
    scope: AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
    compilation_catalog: CompilationCatalog | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    checkpoint()
    with scope:
        checkpoint()
        model = build_model(graph, checkpoint=checkpoint)
        checkpoint()
        compile_model(model, graph, compilation_catalog=compilation_catalog)
        input_values = _dummy_inputs(keras_module, model)
        model_input: Any = input_values[0] if len(input_values) == 1 else input_values
        output_values = model(model_input, training=False)
        checkpoint()

    actual_outputs = (
        list(output_values) if isinstance(output_values, (list, tuple)) else [output_values]
    )
    synchronize(actual_outputs)
    graph_inputs = input_nodes(graph, get_node_catalog())
    graph_outputs = resolved_outputs(graph, get_node_catalog())
    inputs = [
        {
            "node_id": node.id,
            "shape": _concrete_shape(tensor),
            "dtype": _dtype_name(tensor),
        }
        for node, tensor in zip(graph_inputs, input_values, strict=False)
    ]
    outputs = [
        _output_metadata(output, tensor)
        for output, tensor in zip(graph_outputs, actual_outputs, strict=False)
    ]
    return {
        "status": "succeeded",
        "accelerator": capabilities["accelerator"],
        "device_name": capabilities["device_name"],
        "inputs": inputs,
        "outputs": outputs,
        "parameter_count": int(model.count_params()),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "compilation": _configured_compilation_summary(graph, "keras"),
    }

###############################################################################
def _inspection_input_arrays(graph: ModelGraph, *, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    embedding_vocabularies = [
        int(node.parameters.get("vocabulary_size", 0))
        for node in graph.nodes
        if node.definition_id == "layer.embedding"
        and isinstance(node.parameters.get("vocabulary_size"), int)
        and int(node.parameters.get("vocabulary_size", 0)) > 0
    ]
    embedding_upper_bound = min(embedding_vocabularies, default=16)
    for node in input_nodes(graph, get_node_catalog()):
        # ``RuntimeInputParameters.get`` deliberately exposes the runtime
        # batch axis (``[None, *feature_shape]``).  Synthetic inspection
        # inputs own that axis and must therefore validate the persisted
        # feature shape from the underlying mapping instead of rejecting the
        # expected placeholder ``None``.
        shape_value = dict.get(node.parameters, "shape", [1])
        if not isinstance(shape_value, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in shape_value
        ):
            raise ValueError(
                f"dynamic_dimension_unresolved: input node '{node.id}' has no fully resolved feature shape"
            )
        shape = [1, *[int(value) for value in shape_value]]
        dtype = str(node.parameters.get("dtype", "float32"))
        if dtype == "bool":
            values = rng.random(shape) > 0.5
        elif dtype.startswith("int"):
            values = rng.integers(0, embedding_upper_bound, size=shape, dtype=np.int64)
            if dtype == "int32":
                values = values.astype(np.int32)
            elif dtype != "int64":
                raise ValueError(f"unsupported_dtype: input node '{node.id}' uses {dtype}")
        elif dtype in {"float16", "float32", "float64"}:
            values = rng.uniform(-1.0, 1.0, size=shape).astype(dtype)
        else:
            raise ValueError(f"unsupported_dtype: input node '{node.id}' uses {dtype}")
        result.append(values)
    return result

###############################################################################
def run_keras_inspection(
    graph: ModelGraph,
    *,
    keras_module: Any,
    build_model: Callable[..., Any],
    compile_model: Callable[..., Any],
    capabilities: dict[str, object],
    scope: AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
    seed: int,
    project_id: str,
    project_revision: int,
    output_root: Path,
    worker: WorkerConfiguration,
    compilation_catalog: CompilationCatalog | None = None,
) -> dict[str, object]:
    checkpoint()
    with suppress(Exception):
        keras_module.utils.set_random_seed(seed)
    # Keras normally reapplies the model's persisted random seed while it is
    # being built.  Keep that persisted configuration untouched, but make an
    # explicitly requested inspection seed authoritative for this ephemeral
    # execution copy so reruns can be reproduced from the public contract.
    inspection_graph = graph.model_copy(deep=True)
    inspection_graph.model_configuration.keras.random_seed = seed
    trace_outputs: dict[str, dict[str, Any]] = {}
    with scope:
        model = build_model(inspection_graph, checkpoint=checkpoint, trace_outputs=trace_outputs)
        compile_model(model, inspection_graph, compilation_catalog=compilation_catalog)
        arrays = _inspection_input_arrays(inspection_graph, seed=seed)
        # Keep the Keras input structure explicit even for one-input models;
        # passing the bare tensor makes Keras warn that the symbolic input
        # list was flattened, while the values themselves are unchanged.
        model_input: Any = arrays
        probe_values: list[Any] = []
        probe_keys: list[tuple[str, str, int | None]] = []
        for node_id in [node.id for node in graph.nodes]:
            for port_id, symbolic in trace_outputs.get(node_id, {}).items():
                if isinstance(symbolic, (list, tuple)):
                    for item_index, item in enumerate(symbolic):
                        probe_values.append(item)
                        probe_keys.append((node_id, port_id, item_index))
                else:
                    probe_values.append(symbolic)
                    probe_keys.append((node_id, port_id, None))
        if not probe_values:
            raise RuntimeError(
                "capture_unsupported: Keras inspection did not expose any node outputs"
            )
        probe = keras_module.Model(inputs=model.inputs, outputs=probe_values)
        raw_values = probe(model_input, training=False)
        actual_values = list(raw_values) if isinstance(raw_values, (list, tuple)) else [raw_values]
        trace: dict[str, dict[str, Any]] = {}
        for (node_id, port_id, item_index), value in zip(probe_keys, actual_values, strict=True):
            node_trace = trace.setdefault(node_id, {})
            if item_index is None:
                node_trace[port_id] = value
                continue
            # A tensor-list port is represented as one manifest port with
            # explicit item indices. Reassemble the list after the probe
            # model flattens it into individual symbolic outputs.
            items = node_trace.setdefault(port_id, [])
            if not isinstance(items, list):
                raise RuntimeError(
                    f"capture_unsupported: Keras returned mixed values for output port '{port_id}'"
                )
            while len(items) <= item_index:
                items.append(None)
            items[item_index] = value
        synchronize(actual_values)
        result = persist_inspection(
            graph,
            project_id=project_id,
            project_revision=project_revision,
            backend=worker.backend,
            api=worker.api,
            worker=worker,
            seed=seed,
            input_description="Seeded batch-size-1 synthetic tensors: float values in [-1, 1]; integers use a safe non-negative range bounded by embedding vocabularies when applicable; booleans sampled at 0.5.",
            trace=trace,
            inputs=arrays,
            output_root=output_root,
        )
        result.update({"status": "succeeded", "device_name": capabilities.get("device_name", "CPU"), "accelerator": capabilities.get("accelerator", "cpu")})
        return result

###############################################################################
def run_native_inspection(
    graph: ModelGraph,
    *,
    backend: ExecutionBackend,
    capabilities: dict[str, object],
    scope: AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
    seed: int,
    project_id: str,
    project_revision: int,
    output_root: Path,
    worker: WorkerConfiguration,
    compilation_catalog: CompilationCatalog | None = None,
) -> dict[str, object]:
    from models.native_runtime import build_native_model, forward_native_with_trace

    native_configuration = getattr(graph.model_configuration.native, backend.value)
    compiler_enabled = bool(
        native_configuration.compiler.parameters.get(
            "jit_compile" if backend is ExecutionBackend.TENSORFLOW else "enabled",
            False,
        )
    )
    # Instrumentation must observe the same backend without allowing an
    # optimized wrapper to erase the trace contract. Build an isolated copy
    # with only the relevant compiler switch disabled; the persisted graph and
    # its execution profile remain unchanged and the session records the
    # bypass explicitly.
    inspection_graph = graph.model_copy(deep=True)
    if compiler_enabled:
        inspection_configuration = getattr(inspection_graph.model_configuration.native, backend.value)
        parameters = dict(inspection_configuration.compiler.parameters)
        parameters["jit_compile" if backend is ExecutionBackend.TENSORFLOW else "enabled"] = False
        inspection_configuration.compiler = inspection_configuration.compiler.model_copy(
            update={"parameters": parameters}
        )

    raw_arrays = _inspection_input_arrays(graph, seed=seed)
    if backend is ExecutionBackend.PYTORCH:
        import torch

        inputs = [torch.as_tensor(value) for value in raw_arrays]
    elif backend is ExecutionBackend.TENSORFLOW:
        import tensorflow as tf

        inputs = [tf.convert_to_tensor(value) for value in raw_arrays]
    else:
        import jax.numpy as jnp

        inputs = [jnp.asarray(value) for value in raw_arrays]
    with scope:
        handle = build_native_model(
            inspection_graph,
            backend,
            checkpoint=checkpoint,
            compilation_catalog=compilation_catalog,
            seed=seed,
            zero_parameters=False,
            input_values=inputs,
        )
        outputs, trace = forward_native_with_trace(handle)
        synchronize(outputs)
        # Compiled wrappers are optimized for the normal output contract and
        # may not preserve the optional trace payload. The raw generated model
        # still executes on the same backend and carries the same initialized
        # parameters, so expose this observability choice in session metadata.
        compiler_mode = "bypassed_for_inspection" if compiler_enabled else "normal"
        result = persist_inspection(
            graph,
            project_id=project_id,
            project_revision=project_revision,
            backend=backend,
            api=worker.api,
            worker=worker,
            seed=seed,
            input_description="Seeded batch-size-1 synthetic tensors: float values in [-1, 1]; integers use a safe non-negative range bounded by embedding vocabularies when applicable; booleans sampled at 0.5.",
            trace=trace,
            inputs=inputs,
            output_root=output_root,
            compiler_mode=compiler_mode,
        )
        result.update({"status": "succeeded", "device_name": capabilities.get("device_name", "CPU"), "accelerator": capabilities.get("accelerator", "cpu")})
        return result

###############################################################################
def run_native_smoke(
    graph: ModelGraph,
    *,
    backend: ExecutionBackend,
    capabilities: dict[str, object],
    scope: AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
    compilation_catalog: CompilationCatalog | None = None,
) -> dict[str, object]:
    """Run a smoke pass through the raw framework implementation only."""

    from models.native_runtime import build_native_model, forward_native

    started = time.perf_counter()
    checkpoint()
    with scope:
        handle = build_native_model(graph, backend, checkpoint=checkpoint, compilation_catalog=compilation_catalog)
        checkpoint()
        actual_outputs = forward_native(handle)
        compilation = _verify_native_compilation(graph, handle, checkpoint)
        synchronize(actual_outputs)
        checkpoint()
    graph_inputs = input_nodes(graph, get_node_catalog())
    graph_outputs = resolved_outputs(graph, get_node_catalog())
    return {
        "status": "succeeded",
        "accelerator": capabilities.get("accelerator", "cpu"),
        "device_name": capabilities.get("device_name", "CPU"),
        "inputs": [
            {
                "node_id": node.id,
                "shape": _concrete_shape(tensor),
                "dtype": _dtype_name(tensor),
            }
            for node, tensor in zip(graph_inputs, handle.inputs, strict=False)
        ],
        "outputs": [
            _output_metadata(output, tensor)
            for output, tensor in zip(graph_outputs, actual_outputs, strict=False)
        ],
        "parameter_count": handle.parameter_count(),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "compilation": compilation,
        "training": "not_run",
    }

###############################################################################
def _configured_compilation_summary(graph: ModelGraph, api: str) -> dict[str, object]:
    configuration = (
        graph.model_configuration.keras
        if api == "keras"
        else graph.model_configuration.native.pytorch
    )
    output_items = [
        output for output in resolved_outputs(graph, get_node_catalog()) if output.node is not None
    ]

    def selected(values: dict[str, Any], output: Any, default: Any) -> Any:
        value = values.get("__default__", default)
        for selector in output_selectors(output):
            if selector and selector in values:
                value = values[selector]
                break
        return value

    compilation = configuration.compilation

    return {
        "verified": True,
        "optimizer": {
            "id": compilation.optimizer.id,
            "instantiated": True,
        },
        "losses": {
            output.name: _compilation_summary_value(selected(compilation.losses, output, None))
            for output in output_items
            if selected(compilation.losses, output, None) is not None
        },
        "metrics": {
            output.name: _compilation_summary_value(selected(compilation.metrics, output, []))
            for output in output_items
            if selected(compilation.metrics, output, [])
        },
        "training": "not_run",
    }


###############################################################################
def _compilation_summary_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_compilation_summary_value(item) for item in value]
    return value

###############################################################################
def _output_metadata(output: Any, tensor: Any) -> dict[str, object]:
    metadata: dict[str, object] = {
        "node_id": output.binding.node_id,
        "shape": _concrete_shape(tensor),
        "dtype": _dtype_name(tensor),
    }
    if output.binding.port_id != "output" or output.name != output.binding.node_id:
        metadata.update({"port_id": output.binding.port_id, "name": output.name})
    return metadata

###############################################################################
def _verify_native_compilation(
    graph: ModelGraph,
    handle: Any,
    checkpoint: Callable[[], None],
) -> dict[str, object] | None:
    """Instantiate and evaluate generated native PyTorch helpers without training."""

    if handle.compilation is None:
        return None
    if handle.backend is ExecutionBackend.TENSORFLOW:
        return _verify_tensorflow_compilation(graph, handle, checkpoint)
    if handle.backend is ExecutionBackend.JAX:
        return _verify_jax_compilation(graph, handle, checkpoint)
    import torch

    compilation = handle.compilation
    output_bindings = resolved_outputs(graph, get_node_catalog())
    output_values = {
        output.name: value
        for output, value in zip(output_bindings, handle.outputs, strict=True)
    }
    loss_functions = compilation.get("loss_functions", {})
    loss_objects = compilation.get("losses", {})
    metric_functions = compilation.get("metric_functions", {})
    target_contracts = compilation.get("target_contracts", {})
    metric_target_contracts = compilation.get("metric_target_contracts", {})
    loss_summary: dict[str, object] = {}
    metric_summary: dict[str, object] = {}
    with torch.no_grad():
        for output_id, function in loss_functions.items():
            prediction = output_values[output_id]
            contract = str(target_contracts.get(output_id, "continuous_same_shape"))
            value = function(prediction, _synthetic_target(prediction, contract))
            _assert_finite_scalar(value, f"loss for output {output_id}")
            loss_object = loss_objects.get(output_id, {})
            loss_summary[output_id] = {
                "id": loss_object.get("id") if isinstance(loss_object, dict) else None,
                "parameters": (
                    loss_object.get("parameters", {})
                    if isinstance(loss_object, dict)
                    else {}
                ),
                "value": float(value.detach().cpu().item()),
                "target_contract": contract,
            }
            checkpoint()
        for output_id, functions in metric_functions.items():
            prediction = output_values[output_id]
            contract = str(target_contracts.get(output_id, "continuous_same_shape"))
            values: dict[str, object] = {}
            for metric_id, function in functions.items():
                metric_contract = str(
                    metric_target_contracts.get(output_id, {}).get(metric_id, contract)
                )
                value = function(prediction, _synthetic_target(prediction, metric_contract))
                _assert_finite_scalar(value, f"metric {metric_id} for output {output_id}")
                values[metric_id] = float(value.detach().cpu().item())
                checkpoint()
            if values:
                metric_summary[output_id] = values
    optimizer_object = compilation.get("optimizer", {})
    optimizer_runtime = compilation.get("optimizer_runtime")
    return {
        "verified": True,
        "optimizer": {
            "id": optimizer_object.get("id"),
            "instantiated": optimizer_runtime is not None,
        },
        "losses": loss_summary,
        "metrics": metric_summary,
        "training": "not_run",
    }

###############################################################################
def _synthetic_target(prediction: Any, contract: str) -> Any:
    import torch

    if contract == "class_index":
        return torch.zeros(prediction.shape[:-1], dtype=torch.long, device=prediction.device)
    if contract == "one_hot_same_shape":
        target = torch.zeros_like(prediction, dtype=torch.float32)
        target[..., 0] = 1.0
        return target
    if contract == "probability_distribution_same_shape":
        width = max(int(prediction.shape[-1]), 1)
        return torch.full_like(prediction, 1.0 / width, dtype=torch.float32)
    return torch.zeros_like(prediction, dtype=torch.float32)

###############################################################################
def _assert_finite_scalar(value: Any, description: str) -> None:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.numel() != 1
        or not bool(torch.isfinite(value).all())
    ):
        raise RuntimeError(f"Native PyTorch {description} did not return one finite scalar")

###############################################################################
def _verify_tensorflow_compilation(
    graph: ModelGraph,
    handle: Any,
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    import tensorflow as tf

    compilation = handle.compilation
    output_bindings = resolved_outputs(graph, get_node_catalog())
    output_values = {
        output.name: value
        for output, value in zip(output_bindings, handle.outputs, strict=True)
    }
    loss_functions = compilation.get("loss_functions", {})
    loss_objects = compilation.get("losses", {})
    metric_functions = compilation.get("metric_functions", {})
    target_contracts = compilation.get("target_contracts", {})
    metric_target_contracts = compilation.get("metric_target_contracts", {})
    loss_summary: dict[str, object] = {}
    metric_summary: dict[str, object] = {}

    def target_for(prediction: Any, contract: str) -> Any:
        if contract == "class_index":
            return tf.zeros(tf.shape(prediction)[:-1], dtype=tf.int32)
        if contract == "one_hot_same_shape":
            return tf.one_hot(
                tf.zeros(tf.shape(prediction)[:-1], dtype=tf.int32),
                depth=tf.shape(prediction)[-1],
                dtype=tf.float32,
            )
        if contract == "probability_distribution_same_shape":
            width = tf.maximum(tf.shape(prediction)[-1], 1)
            return tf.ones_like(prediction, dtype=tf.float32) / tf.cast(width, tf.float32)
        return tf.zeros_like(prediction, dtype=tf.float32)

    def scalar(value: Any, description: str) -> float:
        tensor = tf.convert_to_tensor(value)
        if tensor.shape.num_elements() not in (None, 1):
            raise RuntimeError(f"Native TensorFlow {description} did not return one scalar")
        if not bool(tf.reduce_all(tf.math.is_finite(tensor))):
            raise RuntimeError(f"Native TensorFlow {description} was not finite")
        return float(tf.reshape(tensor, [-1])[0].numpy())

    for output_id, function in loss_functions.items():
        prediction = output_values[output_id]
        contract = str(target_contracts.get(output_id, "continuous_same_shape"))
        value = scalar(function(prediction, target_for(prediction, contract)), f"loss for output {output_id}")
        loss_object = loss_objects.get(output_id, {})
        loss_summary[output_id] = {
            "id": loss_object.get("id") if isinstance(loss_object, dict) else None,
            "parameters": loss_object.get("parameters", {}) if isinstance(loss_object, dict) else {},
            "value": value,
            "target_contract": contract,
        }
        checkpoint()
    for output_id, functions in metric_functions.items():
        prediction = output_values[output_id]
        contract = str(target_contracts.get(output_id, "continuous_same_shape"))
        values: dict[str, object] = {}
        for metric_id, function in functions.items():
            metric_contract = str(
                metric_target_contracts.get(output_id, {}).get(metric_id, contract)
            )
            values[metric_id] = scalar(
                function(prediction, target_for(prediction, metric_contract)),
                f"metric {metric_id} for output {output_id}",
            )
            checkpoint()
        if values:
            metric_summary[output_id] = values

    optimizer_runtime = compilation.get("optimizer_runtime")
    trainable_variables = list(getattr(handle.model, "trainable_variables", []))
    optimizer_changed = False
    if optimizer_runtime is not None and trainable_variables:
        before = [variable.numpy().copy() for variable in trainable_variables]
        with tf.GradientTape() as tape:
            objective = tf.add_n(
                [tf.reduce_sum(tf.square(variable - 1.0)) for variable in trainable_variables]
            )
        gradients = tape.gradient(objective, trainable_variables)
        optimizer_runtime.apply_gradients(zip(gradients, trainable_variables, strict=True))
        optimizer_changed = any(
            not np.array_equal(previous, variable.numpy())
            for previous, variable in zip(before, trainable_variables, strict=True)
        )
        if not optimizer_changed:
            raise RuntimeError("Native TensorFlow optimizer update did not change a trainable value")
        if not all(np.isfinite(variable.numpy()).all() for variable in trainable_variables):
            raise RuntimeError("Native TensorFlow optimizer update produced a non-finite value")

    optimizer_object = compilation.get("optimizer", {})
    return {
        "verified": True,
        "optimizer": {
            "id": optimizer_object.get("id"),
            "instantiated": optimizer_runtime is not None,
            "updated": optimizer_changed,
        },
        "losses": loss_summary,
        "metrics": metric_summary,
        "training": "not_run",
    }

###############################################################################
def _verify_jax_compilation(
    graph: ModelGraph,
    handle: Any,
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    import optax

    compilation = handle.compilation
    output_bindings = resolved_outputs(graph, get_node_catalog())
    output_values = {
        output.name: value
        for output, value in zip(output_bindings, handle.outputs, strict=True)
    }
    loss_functions = compilation.get("loss_functions", {})
    loss_objects = compilation.get("losses", {})
    metric_functions = compilation.get("metric_functions", {})
    target_contracts = compilation.get("target_contracts", {})
    metric_target_contracts = compilation.get("metric_target_contracts", {})
    loss_summary: dict[str, object] = {}
    metric_summary: dict[str, object] = {}

    def target_for(prediction: Any, contract: str) -> Any:
        if contract == "class_index":
            return jnp.zeros(prediction.shape[:-1], dtype=jnp.int32)
        if contract == "one_hot_same_shape":
            target = jnp.zeros_like(prediction, dtype=jnp.float32)
            return target.at[..., 0].set(1.0)
        if contract == "probability_distribution_same_shape":
            width = max(int(prediction.shape[-1]), 1)
            return jnp.ones_like(prediction, dtype=jnp.float32) / width
        return jnp.zeros_like(prediction, dtype=jnp.float32)

    def scalar(value: Any, description: str) -> float:
        array = jnp.asarray(value)
        if array.size != 1:
            raise RuntimeError(f"Native JAX {description} did not return one scalar")
        if not bool(jnp.all(jnp.isfinite(array))):
            raise RuntimeError(f"Native JAX {description} was not finite")
        return float(array.reshape(-1)[0])

    for output_id, function in loss_functions.items():
        prediction = output_values[output_id]
        contract = str(target_contracts.get(output_id, "continuous_same_shape"))
        value = scalar(function(prediction, target_for(prediction, contract)), f"loss for output {output_id}")
        loss_object = loss_objects.get(output_id, {})
        loss_summary[output_id] = {
            "id": loss_object.get("id") if isinstance(loss_object, dict) else None,
            "parameters": loss_object.get("parameters", {}) if isinstance(loss_object, dict) else {},
            "value": value,
            "target_contract": contract,
        }
        checkpoint()
    for output_id, functions in metric_functions.items():
        prediction = output_values[output_id]
        contract = str(target_contracts.get(output_id, "continuous_same_shape"))
        values: dict[str, object] = {}
        for metric_id, function in functions.items():
            metric_contract = str(
                metric_target_contracts.get(output_id, {}).get(metric_id, contract)
            )
            values[metric_id] = scalar(
                function(prediction, target_for(prediction, metric_contract)),
                f"metric {metric_id} for output {output_id}",
            )
            checkpoint()
        if values:
            metric_summary[output_id] = values

    optimizer_runtime = compilation.get("optimizer_runtime")
    optimizer_state = compilation.get("optimizer_state")
    optimizer_changed = False
    if optimizer_runtime is not None and optimizer_state is not None:
        gradients = jax.tree_util.tree_map(jnp.ones_like, handle.variables)
        updates, _ = optimizer_runtime.update(gradients, optimizer_state, handle.variables)
        updated_variables = optax.apply_updates(handle.variables, updates)
        before = jax.tree_util.tree_leaves(handle.variables)
        after = jax.tree_util.tree_leaves(updated_variables)
        optimizer_changed = any(
            not np.array_equal(np.asarray(previous), np.asarray(current))
            for previous, current in zip(before, after, strict=True)
        )
        if not optimizer_changed:
            raise RuntimeError("Native JAX optimizer update did not change a trainable value")
        if not all(np.isfinite(np.asarray(value)).all() for value in after):
            raise RuntimeError("Native JAX optimizer update produced a non-finite value")

    optimizer_object = compilation.get("optimizer", {})
    return {
        "verified": True,
        "optimizer": {
            "id": optimizer_object.get("id"),
            "instantiated": optimizer_runtime is not None,
            "updated": optimizer_changed,
        },
        "losses": loss_summary,
        "metrics": metric_summary,
        "training": "not_run",
    }

###############################################################################
def run_model_export(
    graph: ModelGraph,
    destination: Path,
    *,
    export_format: ExportFormat,
    options: ExportOptions,
    backend: ExecutionBackend,
    api: RuntimeApi,
    capabilities: dict[str, object],
    scope: AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
    project_id: str | None = None,
    project_revision: int | None = None,
    compilation_catalog: CompilationCatalog | None = None,
) -> dict[str, object]:
    from models.export_runtime import run_model_export as execute_export

    return execute_export(
        graph,
        destination,
        export_format=export_format,
        options=options,
        backend=backend,
        api=api,
        capabilities=capabilities,
        scope=scope,
        synchronize=synchronize,
        checkpoint=checkpoint,
        project_id=project_id,
        project_revision=project_revision,
        compilation_catalog=compilation_catalog,
    )

###############################################################################
def _dummy_inputs(keras_module: Any, model: Any) -> list[Any]:
    return [
        keras_module.ops.zeros(_concrete_shape(tensor), dtype=_dtype_name(tensor))
        for tensor in model.inputs
    ]

###############################################################################
def _concrete_shape(tensor: Any) -> list[int]:
    shape = getattr(tensor, "shape", ())
    result: list[int] = []
    for dimension in shape:
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            value = 1
        result.append(value if value > 0 else 1)
    return result

###############################################################################
def _dtype_name(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", "float32")
    name = str(getattr(dtype, "name", dtype))
    return name.rsplit(".", 1)[-1]

###############################################################################
def empty_memory_metrics() -> dict[str, int | None]:
    return {
        "framework_allocated_bytes": None,
        "framework_reserved_bytes": None,
        "framework_peak_bytes": None,
        "device_used_bytes": None,
        "device_total_bytes": None,
    }
