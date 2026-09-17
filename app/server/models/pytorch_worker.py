# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOperatorIssue=false, reportArgumentType=false

from __future__ import annotations

import contextlib
import gc
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from domain.compilation_catalog import CompilationCatalog
from domain.enums import ExecutionBackend
from domain.exports import ExportFormat, ExportOptions
from domain.graph import ModelGraph
from domain.runtime import MemoryPolicyKind, RuntimeApi, WorkerConfiguration
from models.worker_backend import (
    BackendWorker,
    empty_memory_metrics,
    run_keras_smoke,
    run_keras_inspection,
    run_model_export,
    run_native_inspection,
    run_native_smoke,
)


###############################################################################
class PyTorchWorker:
    backend = ExecutionBackend.PYTORCH

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._config: WorkerConfiguration | None = None

    # -------------------------------------------------------------------------
    def configure(self, config: WorkerConfiguration) -> None:
        self._config = config
        if config.device_id.startswith(("cuda:", "rocm:")) and bool(torch.cuda.is_available()):
            torch.cuda.set_device(0)
            if config.memory_policy.kind is MemoryPolicyKind.PYTORCH_FRACTION:
                fraction = config.memory_policy.fraction
                if fraction is None:
                    raise RuntimeError("pytorch_fraction requires a fraction")
                torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)

    # -------------------------------------------------------------------------
    def capabilities(self) -> dict[str, object]:
        requested_device = self._config.device_id if self._config is not None else "auto"
        accelerator = "cpu"
        device_name = "CPU"
        if (requested_device == "auto" or requested_device.startswith(("cuda:", "rocm:"))) and bool(
            torch.cuda.is_available()
        ):
            accelerator = "cuda"
            device_name = str(torch.cuda.get_device_name(0))
        elif (
            requested_device.startswith("xpu:")
            and hasattr(torch, "xpu")
            and bool(torch.xpu.is_available())
        ):
            accelerator = "xpu"
            device_name = str(torch.xpu.get_device_name(0))
        elif (
            requested_device.startswith("mps:")
            and hasattr(torch.backends, "mps")
            and bool(torch.backends.mps.is_available())
        ):
            accelerator = "mps"
            device_name = "Apple Metal Performance Shaders"
        backend_name = "pytorch"
        if self._config is not None and self._config.api is RuntimeApi.KERAS:
            import keras

            backend_name = str(keras.backend.backend())
        return {
            "backend": backend_name,
            "available": True,
            "accelerator": accelerator,
            "device_name": device_name,
        }

    # -------------------------------------------------------------------------
    def native_compilation_capabilities(self) -> dict[str, bool]:
        available = callable(getattr(torch, "compile", None))
        if available and sys.platform == "win32":
            # PyTorch's default Inductor backend emits C++ on Windows. The
            # callable API alone is not enough: without the MSVC developer
            # environment, the first compiled graph fails with InvalidCxxCompiler.
            available = shutil.which("cl") is not None
        return {"torch_compile": available}

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
        if bool(torch.cuda.is_available()):
            for index in range(int(torch.cuda.device_count())):
                devices.append(
                    {
                        "device_id": f"cuda:{index}",
                        "lease_key": f"cuda:{index}",
                        "kind": "cuda",
                        "index": index,
                        "name": str(torch.cuda.get_device_name(index)),
                        "accelerator": "cuda",
                        "total_memory_bytes": int(
                            torch.cuda.get_device_properties(index).total_memory
                        ),
                        "supported_memory_policies": [
                            MemoryPolicyKind.DEFAULT.value,
                            MemoryPolicyKind.PYTORCH_FRACTION.value,
                        ],
                    }
                )
        if hasattr(torch, "xpu") and bool(torch.xpu.is_available()):
            devices.append(
                {
                    "device_id": "xpu:0",
                    "lease_key": "xpu:0",
                    "kind": "xpu",
                    "index": 0,
                    "name": str(torch.xpu.get_device_name(0)),
                    "accelerator": "xpu",
                    "supported_memory_policies": [MemoryPolicyKind.DEFAULT.value],
                }
            )
        if hasattr(torch.backends, "mps") and bool(torch.backends.mps.is_available()):
            devices.append(
                {
                    "device_id": "mps:0",
                    "lease_key": "mps:0",
                    "kind": "mps",
                    "index": 0,
                    "name": "Apple Metal Performance Shaders",
                    "accelerator": "mps",
                    "supported_memory_policies": [MemoryPolicyKind.DEFAULT.value],
                }
            )
        return devices

    # -------------------------------------------------------------------------
    def memory_metrics(self) -> dict[str, int | None]:
        metrics = empty_memory_metrics()
        try:
            if self.capabilities().get("accelerator") == "cuda":
                metrics["framework_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
                metrics["framework_reserved_bytes"] = int(torch.cuda.memory_reserved(0))
                metrics["framework_peak_bytes"] = int(torch.cuda.max_memory_allocated(0))
                device_memory_used = getattr(torch.cuda, "device_memory_used", None)
                if callable(device_memory_used):
                    metrics["device_used_bytes"] = int(device_memory_used(0))
                metrics["device_total_bytes"] = int(
                    torch.cuda.get_device_properties(0).total_memory
                )
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            return metrics
        return metrics

    # -------------------------------------------------------------------------
    def device_scope(self) -> contextlib.AbstractContextManager[Any]:
        capabilities = self.capabilities()
        if capabilities.get("accelerator") in {"cuda", "xpu", "mps"}:
            if self._config is not None and self._config.api is RuntimeApi.NATIVE:
                device = (
                    "cuda:0"
                    if capabilities.get("accelerator") == "cuda"
                    else str(capabilities["accelerator"]) + ":0"
                )
                return torch.device(device)
            import keras

            return keras.device("gpu:0")
        return contextlib.nullcontext()

    # -------------------------------------------------------------------------
    def synchronize(self, values: list[Any]) -> None:
        for value in values:
            block_until_ready = getattr(value, "block_until_ready", None)
            if callable(block_until_ready):
                block_until_ready()
        try:
            if self.capabilities().get("accelerator") == "cuda":
                torch.cuda.synchronize()
        except (RuntimeError, TypeError):
            pass

    # -------------------------------------------------------------------------
    def run_smoke(
        self,
        graph: ModelGraph,
        checkpoint: Callable[[], None],
        *,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> dict[str, object]:
        if self._config is not None and self._config.api is RuntimeApi.NATIVE:
            return run_native_smoke(
                graph,
                backend=self.backend,
                capabilities=self.capabilities(),
                scope=self.device_scope(),
                synchronize=self.synchronize,
                checkpoint=checkpoint,
                compilation_catalog=compilation_catalog,
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
            compilation_catalog=compilation_catalog,
        )

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
    ) -> dict[str, object]:
        if self._config is not None and self._config.api is RuntimeApi.NATIVE:
            return run_native_inspection(
                graph,
                backend=self.backend,
                capabilities=self.capabilities(),
                scope=self.device_scope(),
                synchronize=self.synchronize,
                checkpoint=checkpoint,
                seed=seed,
                project_id=project_id,
                project_revision=project_revision,
                output_root=output_root,
                worker=self._config,
                compilation_catalog=compilation_catalog,
            )
        import keras

        from models import keras_worker
        from models.keras_runtime import compile_model

        return run_keras_inspection(
            graph,
            keras_module=keras,
            build_model=keras_worker.build_model,
            compile_model=compile_model,
            capabilities=self.capabilities(),
            scope=self.device_scope(),
            synchronize=self.synchronize,
            checkpoint=checkpoint,
            seed=seed,
            project_id=project_id,
            project_revision=project_revision,
            output_root=output_root,
            worker=self._config or WorkerConfiguration.model_validate({"api": "keras", "backend": self.backend.value, "device_id": "cpu", "memory_policy": {"kind": "default"}, "precision": "float32"}),
            compilation_catalog=compilation_catalog,
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
        compilation_catalog: CompilationCatalog | None = None,
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
            compilation_catalog=compilation_catalog,
        )

    # -------------------------------------------------------------------------
    def cleanup(self) -> None:
        try:
            if self.capabilities().get("accelerator") == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self._config is not None and self._config.api is RuntimeApi.KERAS:
                import keras

                keras.backend.clear_session()
        except (AttributeError, RuntimeError, TypeError):
            pass
        gc.collect()

###############################################################################
def create_worker() -> BackendWorker:
    return PyTorchWorker()
