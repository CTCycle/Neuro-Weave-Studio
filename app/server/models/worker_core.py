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
from domain.compilation_catalog import CompilationCatalog, get_compilation_catalog
from domain.exports import ExportFormat, ExportOptions, export_suffix
from domain.graph import ModelGraph
from domain.runtime import WorkerConfiguration
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
        for attempt in range(20):
            try:
                os.replace(temporary, self.status_file)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                # Windows can briefly keep the controller's status read open.
                time.sleep(0.01)

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
    context.checkpoint("preflight", "Configuring the selected backend", 5)
    worker.configure(context.config)
    context.checkpoint(
        "building",
        "Importing the selected framework API and applying the device policy",
        15,
    )
    operation = cast(dict[str, object], payload["operation"])
    operation_type = str(operation.get("type"))
    graph = ModelGraph.model_validate(operation.get("graph"))
    compilation_catalog = CompilationCatalog.model_validate(
        operation.get("compilation_catalog") or get_compilation_catalog().model_dump(mode="json")
    )
    model_checkpoint = partial(
        context.checkpoint,
        "model_loading",
        "Building the model",
        35,
    )

    if operation_type == "smoke_inference":
        result = worker.run_smoke(graph, model_checkpoint, compilation_catalog=compilation_catalog)
        context.checkpoint("validating", "Forward pass completed", 90)
    elif operation_type == "model_export":
        export_format = ExportFormat(str(operation.get("format")))
        options = ExportOptions.model_validate(operation.get("options") or {})
        artifact_filename = str(operation.get("artifact_filename") or "")
        if not artifact_filename:
            stem = str(operation.get("filename_stem") or "model")
            artifact_filename = f"{stem}{export_suffix(export_format, options)}"
        raw_project_revision = operation.get("project_revision")
        project_revision = raw_project_revision if isinstance(raw_project_revision, int) else None
        context.checkpoint("serializing", f"Serializing {export_format.value} artifact", 60)
        result = worker.run_export(
            graph,
            context.job_dir / artifact_filename,
            export_format,
            options,
            model_checkpoint,
            project_id=str(operation.get("project_id")) if operation.get("project_id") else None,
            project_revision=project_revision,
            compilation_catalog=compilation_catalog,
        )
        context.checkpoint("validating", "Artifact verification completed", 95)
    else:
        raise RuntimeError(f"Unsupported operation '{operation_type}'")

    result.update(worker.memory_metrics())
    return result
