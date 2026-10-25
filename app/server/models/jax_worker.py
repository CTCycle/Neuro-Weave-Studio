# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOperatorIssue=false, reportArgumentType=false

from __future__ import annotations

import contextlib
import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax

from domain.compilation_catalog import CompilationCatalog
from domain.enums import ExecutionBackend
from domain.exports import ExportFormat, ExportOptions
from domain.graph import ModelGraph
from domain.runtime import MemoryPolicyKind, RuntimeApi, WorkerConfiguration
from models.worker_backend import (
    BackendWorker,
    empty_memory_metrics,
    run_keras_inspection,
    run_keras_smoke,
    run_model_export,
    run_native_inspection,
    run_native_smoke,
)


###############################################################################
class JaxWorker:
    backend = ExecutionBackend.JAX

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._config: WorkerConfiguration | None = None

    # -------------------------------------------------------------------------
    def configure(self, config: WorkerConfiguration) -> None:
        self._config = config
        list(jax.devices())

    # -------------------------------------------------------------------------
    def capabilities(self) -> dict[str, object]:
        devices = list(jax.devices())
        accelerator = "cpu"
        device_name = "CPU"
        if devices:
            platform = str(devices[0].platform)
            if platform not in {"cpu", ""}:
                accelerator = platform
            device_name = str(devices[0])
        backend_name = "jax"
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
        return {"jax_jit": callable(getattr(jax, "jit", None))}

    # -------------------------------------------------------------------------
    def device_inventory(self) -> list[dict[str, object]]:
        devices = list(jax.devices())
        inventory: list[dict[str, object]] = []
        for index, device in enumerate(devices):
            platform = str(device.platform)
            if platform == "cpu":
                device_id = "cpu"
                accelerator = "cpu"
            else:
                accelerator = (
                    "rocm"
                    if "rocm" in platform
                    else "cuda"
                    if platform in {"gpu", "cuda"}
                    else platform
                )
                device_id = f"{accelerator}:{index}"
            inventory.append(
                {
                    "device_id": device_id,
                    "lease_key": device_id,
                    "kind": platform,
                    "index": None if platform == "cpu" else index,
                    "name": str(device),
                    "accelerator": accelerator,
                    "supported_memory_policies": [
                        MemoryPolicyKind.DEFAULT.value,
                        MemoryPolicyKind.JAX_NO_PREALLOCATION.value,
                        MemoryPolicyKind.JAX_PREALLOCATE_FRACTION.value,
                        MemoryPolicyKind.JAX_PLATFORM_ALLOCATOR.value,
                    ],
                }
            )
        return inventory

    # -------------------------------------------------------------------------
    def memory_metrics(self) -> dict[str, int | None]:
        metrics = empty_memory_metrics()
        try:
            devices = list(jax.devices())
            if devices and hasattr(devices[0], "memory_stats"):
                stats = cast(dict[str, Any], devices[0].memory_stats() or {})
                metrics["framework_allocated_bytes"] = int(stats.get("bytes_in_use", 0) or 0)
                metrics["framework_reserved_bytes"] = int(stats.get("bytes_reserved", 0) or 0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return metrics
        return metrics

    # -------------------------------------------------------------------------
    def device_scope(self) -> contextlib.AbstractContextManager[Any]:
        capabilities = self.capabilities()
        if capabilities.get("accelerator") not in {None, "cpu"}:
            if self._config is not None and self._config.api is RuntimeApi.NATIVE:
                return jax.default_device(jax.devices()[0])
            import keras

            return keras.device("gpu:0")
        return contextlib.nullcontext()

    # -------------------------------------------------------------------------
    def synchronize(self, values: list[Any]) -> None:
        for value in values:
            block_until_ready = getattr(value, "block_until_ready", None)
            if callable(block_until_ready):
                block_until_ready()

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
                graph, backend=self.backend, capabilities=self.capabilities(), scope=self.device_scope(),
                synchronize=self.synchronize, checkpoint=checkpoint, seed=seed, project_id=project_id,
                project_revision=project_revision, output_root=output_root, worker=self._config,
                compilation_catalog=compilation_catalog,
            )
        import keras

        from models import keras_worker
        from models.keras_runtime import compile_model
        return run_keras_inspection(
            graph, keras_module=keras, build_model=keras_worker.build_model, compile_model=compile_model,
            capabilities=self.capabilities(), scope=self.device_scope(), synchronize=self.synchronize,
            checkpoint=checkpoint, seed=seed, project_id=project_id, project_revision=project_revision,
            output_root=output_root, worker=self._config, compilation_catalog=compilation_catalog,
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
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
            if self._config is not None and self._config.api is RuntimeApi.KERAS:
                import keras

                keras.backend.clear_session()
        except (AttributeError, RuntimeError, TypeError):
            pass
        gc.collect()

###############################################################################
def create_worker() -> BackendWorker:
    return JaxWorker()
