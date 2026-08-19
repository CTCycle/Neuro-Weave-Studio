"""Small one-shot test helper backed by :class:`RuntimeController`.

The application never uses this synchronous convenience surface.  It remains
for focused backend/export tests while all public API work goes through the
observable job controller.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast
from uuid import uuid4

from configurations.settings import get_settings
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from domain.jobs import (
    JobResponse,
    JobSubmitRequest,
    KerasExportOperation,
    SmokeInferenceOperation,
)
from domain.runtime import WorkerConfiguration
from models.runtime_controller import RuntimeController

###############################################################################
class KerasBackendRunner:
    """Test-only adapter that waits on the new process-isolated job controller."""

    def __init__(self) -> None:
        self._controller = RuntimeController(get_settings())
        self._controller.probe_statuses()

    # -------------------------------------------------------------------------
    def available_backends(self) -> tuple[ExecutionBackend, ...]:
        return tuple(
            backend
            for backend, status in self._controller.backend_statuses.items()
            if status.available
        )

    # -------------------------------------------------------------------------
    def probe(self, backend: ExecutionBackend) -> bool:
        status = self._controller.backend_statuses.get(backend)
        return bool(status and status.available)

    # -------------------------------------------------------------------------
    def smoke_test(self, graph: ModelGraph, backend: ExecutionBackend) -> dict[str, object]:
        job = self._submit_smoke(graph, backend)
        result = self._wait(job.id)
        if result.state.value != "succeeded":
            raise RuntimeError(result.error.message if result.error else "Keras inference worker failed")
        return cast(dict[str, object], result.result or {})

    # -------------------------------------------------------------------------
    def export(self, graph: ModelGraph, destination: Path, backend: ExecutionBackend) -> Path:
        job = self._controller.submit(
            JobSubmitRequest(
                operation=KerasExportOperation(project_id=uuid4()),
                worker=self._worker(backend),
            ),
            graph=graph,
            artifact_destination=destination,
            artifact_filename=destination.name,
        )
        result = self._wait(job.id)
        if result.state.value != "succeeded":
            raise RuntimeError(result.error.message if result.error else "Keras export worker failed")
        return destination

    # -------------------------------------------------------------------------
    def close(self) -> None:
        self._controller.shutdown()

    # -------------------------------------------------------------------------
    def _submit_smoke(self, graph: ModelGraph, backend: ExecutionBackend) -> JobResponse:
        return self._controller.submit(
            JobSubmitRequest(
                operation=SmokeInferenceOperation(graph=graph),
                worker=self._worker(backend),
            ),
            graph=graph,
        )

    # -------------------------------------------------------------------------
    def _worker(self, backend: ExecutionBackend) -> WorkerConfiguration:
        status = self._controller.backend_statuses.get(backend)
        if status is None or not status.devices:
            raise RuntimeError(f"Execution backend '{backend.value}' is unavailable")
        return WorkerConfiguration(
            backend=backend,
            device_id=status.devices[0].device_id,
        )

    # -------------------------------------------------------------------------
    def _wait(self, job_id: str) -> JobResponse:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            result = self._controller.get(job_id)
            if result.state.value in {"succeeded", "terminated", "failed"}:
                return result
            time.sleep(0.05)
        self._controller.stop(job_id)
        raise TimeoutError("Keras worker job timed out")
