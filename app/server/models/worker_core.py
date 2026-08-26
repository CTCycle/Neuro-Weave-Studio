# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any, cast

from common.process_control import process_rss_bytes
from domain.graph import ModelGraph
from domain.runtime import RuntimeApi, WorkerConfiguration
from models.worker_backend import BackendWorker


###############################################################################
class WorkerCancelled(RuntimeError):
    pass

###############################################################################
class WorkerContext:

    # -------------------------------------------------------------------------
    def __init__(self, job_dir: Path, config: WorkerConfiguration) -> None:
        self.job_dir = job_dir
        self.config = config
        self.cancel_file = job_dir / "cancel.requested"
        self.status_file = job_dir / "status.json"

    # -------------------------------------------------------------------------
    def checkpoint(self, phase: str, message: str, percent: int | None = None) -> None:
        self.write_status(
            phase=phase,
            message=message,
            percent=percent,
            state="running",
        )
        if self.cancel_file.exists():
            self.write_status(
                phase="cancelling",
                message="Cancellation observed at a safe checkpoint",
                percent=percent,
                state="stop_requested",
            )
            raise WorkerCancelled("Worker cancellation requested")

    # -------------------------------------------------------------------------
    def write_status(
        self,
        *,
        phase: str,
        message: str,
        state: str,
        percent: int | None = None,
        **extra: object,
    ) -> None:
        payload: dict[str, object] = {
            "state": state,
            "phase": phase,
            "message": message,
            "percent": percent,
            "pid": os.getpid(),
            "updated_at": time.time(),
            "process_rss_bytes": process_rss_bytes(os.getpid()),
            "backend": self.config.backend.value,
            "api": self.config.api.value,
            "device_id": self.config.device_id,
            **extra,
        }
        temporary = self.status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.status_file)

###############################################################################
def read_request(job_dir: Path) -> tuple[WorkerContext, dict[str, Any]]:
    payload = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    with suppress(OSError):
        (job_dir / "request.json").unlink()
    config = WorkerConfiguration.model_validate(payload["worker"])
    return WorkerContext(job_dir, config), cast(dict[str, Any], payload)

###############################################################################
def run_job(
    context: WorkerContext,
    payload: dict[str, Any],
    worker: BackendWorker,
) -> dict[str, object]:
    if context.config.api is not RuntimeApi.KERAS:
        raise RuntimeError("Native framework inference is not implemented")
    context.checkpoint("importing", "Configuring the selected backend", 5)
    worker.configure(context.config)
    context.checkpoint("importing", "Importing Keras and the selected backend", 15)
    context.checkpoint("initializing", "Applying device and memory policy", 25)
    operation = cast(dict[str, object], payload["operation"])
    operation_type = str(operation.get("type"))
    graph = ModelGraph.model_validate(operation.get("graph"))
    model_checkpoint = partial(
        context.checkpoint,
        "model_loading",
        "Building the model",
        35,
    )

    if operation_type == "smoke_inference":
        result = worker.run_smoke(graph, model_checkpoint)
        context.checkpoint("publishing", "Forward pass completed", 90)
    elif operation_type == "keras_export":
        result = worker.run_export(
            graph,
            context.job_dir / "artifact.keras",
            model_checkpoint,
        )
        context.checkpoint("publishing", "Artifact verification completed", 95)
    else:
        raise RuntimeError(f"Unsupported operation '{operation_type}'")

    result.update(worker.memory_metrics())
    return result
