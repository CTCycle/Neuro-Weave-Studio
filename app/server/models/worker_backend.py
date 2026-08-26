# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOperatorIssue=false

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, cast

from domain.enums import ExecutionBackend, keras_backend_value
from domain.graph import ModelGraph
from domain.runtime import MemoryPolicyKind, RuntimeApi, WorkerConfiguration


###############################################################################
class BackendWorker(Protocol):
    """Common contract implemented by one isolated execution-backend worker."""

    backend: ExecutionBackend

    def configure(self, config: WorkerConfiguration) -> None: ...

    def capabilities(self) -> dict[str, object]: ...

    def device_inventory(self) -> list[dict[str, object]]: ...

    def memory_metrics(self) -> dict[str, int | None]: ...

    def device_scope(self) -> AbstractContextManager[Any]: ...

    def synchronize(self, values: list[Any]) -> None: ...

    def run_smoke(
        self,
        graph: ModelGraph,
        checkpoint: Callable[[], None],
    ) -> dict[str, object]: ...

    def run_export(
        self,
        graph: ModelGraph,
        destination: Path,
        checkpoint: Callable[[], None],
    ) -> dict[str, object]: ...

    def cleanup(self) -> None: ...


_BACKEND_MODULES = {
    ExecutionBackend.TENSORFLOW: "models.tensorflow_worker",
    ExecutionBackend.PYTORCH: "models.pytorch_worker",
    ExecutionBackend.JAX: "models.jax_worker",
}

###############################################################################
def configure_startup_environment(config: WorkerConfiguration) -> None:
    """Set process-wide visibility before the selected backend module is loaded."""

    if config.api is RuntimeApi.NATIVE:
        raise RuntimeError("Native framework inference is not implemented")
    os.environ["KERAS_BACKEND"] = keras_backend_value(config.backend)

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
) -> dict[str, object]:
    started = time.perf_counter()
    checkpoint()
    with scope:
        checkpoint()
        model = build_model(graph, checkpoint=checkpoint)
        checkpoint()
        compile_model(model, graph)
        input_values = _dummy_inputs(keras_module, model)
        model_input: Any = input_values[0] if len(input_values) == 1 else input_values
        output_values = model(model_input, training=False)
        checkpoint()

    actual_outputs = (
        list(output_values) if isinstance(output_values, (list, tuple)) else [output_values]
    )
    synchronize(actual_outputs)
    input_nodes = [node for node in graph.nodes if node.definition_id.startswith("input.")]
    output_nodes = [node for node in graph.nodes if node.definition_id.startswith("output.")]
    inputs = [
        {
            "node_id": node.id,
            "shape": _concrete_shape(tensor),
            "dtype": _dtype_name(tensor),
        }
        for node, tensor in zip(input_nodes, input_values, strict=False)
    ]
    outputs = [
        {
            "node_id": node.id,
            "shape": _concrete_shape(tensor),
            "dtype": _dtype_name(tensor),
        }
        for node, tensor in zip(output_nodes, actual_outputs, strict=False)
    ]
    return {
        "status": "succeeded",
        "accelerator": capabilities["accelerator"],
        "device_name": capabilities["device_name"],
        "inputs": inputs,
        "outputs": outputs,
        "parameter_count": int(model.count_params()),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }

###############################################################################
def run_keras_export(
    graph: ModelGraph,
    destination: Path,
    *,
    keras_module: Any,
    build_model: Callable[..., Any],
    compile_model: Callable[..., Any],
    capabilities: dict[str, object],
    scope: AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    checkpoint()
    with scope:
        model = build_model(graph, checkpoint=checkpoint)
        checkpoint()
        compile_model(model, graph)
        checkpoint()
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save(destination)
        checkpoint()
        _verify_saved_model(
            destination,
            expected_output_count=len(model.outputs),
            keras_module=keras_module,
            synchronize=synchronize,
            checkpoint=checkpoint,
        )
    return {
        "artifact_name": destination.name,
        "parameter_count": int(model.count_params()),
        "accelerator": capabilities.get("accelerator", "cpu"),
        "device_name": capabilities.get("device_name", "CPU"),
    }

###############################################################################
def _dummy_inputs(keras_module: Any, model: Any) -> list[Any]:
    return [
        keras_module.ops.zeros(_concrete_shape(tensor), dtype=_dtype_name(tensor))
        for tensor in model.inputs
    ]

###############################################################################
def _forward(keras_module: Any, model: Any) -> list[Any]:
    input_values = _dummy_inputs(keras_module, model)
    model_input: Any = input_values[0] if len(input_values) == 1 else input_values
    output_values = model(model_input, training=False)
    return list(output_values) if isinstance(output_values, (list, tuple)) else [output_values]

###############################################################################
def _verify_saved_model(
    destination: Path,
    *,
    expected_output_count: int,
    keras_module: Any,
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
) -> None:
    checkpoint()
    loaded = keras_module.saving.load_model(destination, compile=True)
    checkpoint()
    actual_outputs = _forward(keras_module, loaded)
    synchronize(actual_outputs)
    checkpoint()
    if len(actual_outputs) != expected_output_count:
        raise RuntimeError("Reloaded Keras model returned an unexpected output count")

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
