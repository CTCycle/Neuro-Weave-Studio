# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOperatorIssue=false

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, cast

from common.catalogs.loader import get_node_catalog
from domain.enums import ExecutionBackend, keras_backend_value
from domain.exports import ExportFormat, ExportOptions
from domain.graph import ModelGraph
from domain.runtime import MemoryPolicyKind, RuntimeApi, WorkerConfiguration
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
def run_native_smoke(
    graph: ModelGraph,
    *,
    backend: ExecutionBackend,
    capabilities: dict[str, object],
    scope: AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    """Run a smoke pass through the raw framework implementation only."""

    from models.native_runtime import build_native_model, forward_native

    started = time.perf_counter()
    checkpoint()
    with scope:
        handle = build_native_model(graph, backend, checkpoint=checkpoint)
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

    return {
        "verified": True,
        "optimizer": {
            "id": configuration.optimizer,
            "instantiated": True,
        },
        "losses": {
            output.name: selected(configuration.losses, output, None)
            for output in output_items
            if selected(configuration.losses, output, None) is not None
        },
        "metrics": {
            output.name: selected(configuration.metrics, output, [])
            for output in output_items
            if selected(configuration.metrics, output, [])
        },
        "training": "not_run",
    }


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
    import torch

    compilation = handle.compilation
    output_bindings = resolved_outputs(graph, get_node_catalog())
    output_values = {
        output.name: value
        for output, value in zip(output_bindings, handle.outputs, strict=True)
    }
    loss_functions = compilation.get("losses", {})
    loss_ids = compilation.get("loss_ids", {})
    metric_functions = compilation.get("metrics", {})
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
            loss_summary[output_id] = {
                "id": loss_ids.get(output_id),
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
    optimizer = compilation.get("optimizer")
    return {
        "verified": True,
        "optimizer": {
            "id": compilation.get("optimizer_id"),
            "instantiated": optimizer is not None,
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
