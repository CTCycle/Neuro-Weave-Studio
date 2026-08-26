# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOperatorIssue=false, reportArgumentType=false

from __future__ import annotations

import contextlib
import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tensorflow

from domain.enums import ExecutionBackend
from domain.exports import ExportFormat, ExportOptions
from domain.graph import ModelGraph
from domain.runtime import MemoryPolicyKind, RuntimeApi, WorkerConfiguration
from models.worker_backend import (
    BackendWorker,
    empty_memory_metrics,
    run_keras_smoke,
    run_model_export,
    run_native_smoke,
)


###############################################################################
class TensorFlowWorker:
    backend = ExecutionBackend.TENSORFLOW

    def __init__(self) -> None:
        self._config: WorkerConfiguration | None = None

    # -------------------------------------------------------------------------
    def configure(self, config: WorkerConfiguration) -> None:
        self._config = config
        gpus = list(tensorflow.config.list_physical_devices("GPU"))
        if config.device_id == "cpu":
            tensorflow.config.set_visible_devices([], "GPU")
        elif gpus and config.device_id != "auto":
            visible_gpu = gpus[0]
            tensorflow.config.set_visible_devices(visible_gpu, "GPU")
            policy = config.memory_policy.kind
            if policy is MemoryPolicyKind.TENSORFLOW_GROWTH:
                tensorflow.config.experimental.set_memory_growth(visible_gpu, True)
            elif policy is MemoryPolicyKind.TENSORFLOW_LIMIT_MB:
                tensorflow.config.set_logical_device_configuration(
                    visible_gpu,
                    [
                        tensorflow.config.LogicalDeviceConfiguration(
                            memory_limit=config.memory_policy.limit_mb
                        )
                    ],
                )

    # -------------------------------------------------------------------------
    def capabilities(self) -> dict[str, object]:
        devices = (
            []
            if self._config is not None and self._config.device_id == "cpu"
            else tensorflow.config.list_logical_devices("GPU")
        )
        backend_name = "tensorflow"
        if self._config is not None and self._config.api is RuntimeApi.KERAS:
            import keras

            backend_name = str(keras.backend.backend())
        return {
            "backend": backend_name,
            "available": True,
            "accelerator": "cuda" if devices else "cpu",
            "device_name": str(devices[0].name) if devices else "CPU",
        }

    # -------------------------------------------------------------------------
    def device_inventory(self) -> list[dict[str, object]]:
        devices: list[dict[str, object]] = [
            {
                "device_id": "cpu",
                "lease_key": "cpu",
                "kind": "cpu",
                "index": None,
                "name": "CPU",
                "accelerator": "cpu",
                "supported_memory_policies": [MemoryPolicyKind.DEFAULT.value],
            }
        ]
        for index, device in enumerate(tensorflow.config.list_physical_devices("GPU")):
            devices.append(
                {
                    "device_id": f"cuda:{index}",
                    "lease_key": f"cuda:{index}",
                    "kind": "cuda",
                    "index": index,
                    "name": str(device.name),
                    "accelerator": "cuda",
                    "supported_memory_policies": [
                        MemoryPolicyKind.DEFAULT.value,
                        MemoryPolicyKind.TENSORFLOW_GROWTH.value,
                        MemoryPolicyKind.TENSORFLOW_LIMIT_MB.value,
                    ],
                }
            )
        return devices

    # -------------------------------------------------------------------------
    def memory_metrics(self) -> dict[str, int | None]:
        metrics = empty_memory_metrics()
        try:
            if self.capabilities().get("accelerator") == "cuda":
                info = tensorflow.config.experimental.get_memory_info("GPU:0")
                metrics["framework_allocated_bytes"] = int(info.get("current", 0))
                metrics["framework_peak_bytes"] = int(info.get("peak", 0))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return metrics
        return metrics

    # -------------------------------------------------------------------------
    def device_scope(self) -> contextlib.AbstractContextManager[Any]:
        capabilities = self.capabilities()
        if capabilities.get("accelerator") == "cuda":
            if self._config is not None and self._config.api is RuntimeApi.NATIVE:
                return tensorflow.device("/GPU:0")
            import keras

            return keras.device("gpu:0")
        return contextlib.nullcontext()

    # -------------------------------------------------------------------------
    def synchronize(self, values: list[Any]) -> None:
        del values

    # -------------------------------------------------------------------------
    def run_smoke(
        self,
        graph: ModelGraph,
        checkpoint: Callable[[], None],
    ) -> dict[str, object]:
        if self._config is not None and self._config.api is RuntimeApi.NATIVE:
            return run_native_smoke(
                graph,
                backend=self.backend,
                capabilities=self.capabilities(),
                scope=self.device_scope(),
                synchronize=self.synchronize,
                checkpoint=checkpoint,
            )
        import keras

        from models import keras_worker
        from models.keras_runtime import compile_model

        return run_keras_smoke(
            graph,
            keras_module=keras,
            build_model=keras_worker.build_model,
            compile_model=compile_model,
            capabilities=self.capabilities(),
            scope=self.device_scope(),
            synchronize=self.synchronize,
            checkpoint=checkpoint,
        )

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
    ) -> dict[str, object]:
        return run_model_export(
            graph,
            destination,
            export_format=export_format,
            options=options,
            backend=self.backend,
            api=self._config.api if self._config is not None else RuntimeApi.KERAS,
            capabilities=self.capabilities(),
            scope=self.device_scope(),
            synchronize=self.synchronize,
            checkpoint=checkpoint,
            project_id=project_id,
            project_revision=project_revision,
        )

    # -------------------------------------------------------------------------
    def cleanup(self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError, TypeError):
            import keras

            keras.backend.clear_session()
        gc.collect()

###############################################################################
def create_worker() -> BackendWorker:
    return TensorFlowWorker()
