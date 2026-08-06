from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from domain.enums import ExecutionBackend, keras_backend_value
from domain.graph import ModelGraph

LOGGER = logging.getLogger(__name__)


class KerasBackendRunner:
    """Probe and execute Keras in isolated workers with explicit provenance."""

    def __init__(self) -> None:
        self._server_root = Path(__file__).resolve().parents[1]

    def available_backends(self) -> tuple[ExecutionBackend, ...]:
        return tuple(
            backend for backend, status in self.probe_statuses().items() if status["available"]
        )

    def probe_statuses(self) -> dict[ExecutionBackend, dict[str, object]]:
        statuses: dict[ExecutionBackend, dict[str, object]] = {}
        for backend in ExecutionBackend:
            started = time.perf_counter()
            try:
                result = self._run_worker(backend, ("--probe",), input_text=None)
                expected = keras_backend_value(backend)
                available = result.returncode == 0 and result.stdout.strip() == expected
                reason = None if available else self._reason_code(result)
            except (OSError, subprocess.SubprocessError):
                available = False
                reason = "worker_failed"
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            statuses[backend] = {"available": available, "reason_code": reason}
            LOGGER.info(
                "backend_probe requested=%s worker_backend=%s duration_ms=%s outcome=%s",
                backend.value,
                keras_backend_value(backend),
                duration_ms,
                "available" if available else reason,
            )
        return statuses

    def probe(self, backend: ExecutionBackend) -> bool:
        return bool(self.probe_statuses()[backend]["available"])

    def export(self, graph: ModelGraph, destination: Path, backend: ExecutionBackend) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = self._run_worker(
            backend,
            ("--destination", str(destination)),
            input_text=graph.model_dump_json(),
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "Keras worker failed without a diagnostic"
            raise RuntimeError(detail)
        if not destination.is_file():
            raise RuntimeError("Keras worker completed without creating the model artifact")
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        LOGGER.info(
            "backend_execution requested=%s worker_backend=%s duration_ms=%s outcome=exported",
            backend.value,
            keras_backend_value(backend),
            duration_ms,
        )
        return destination

    @staticmethod
    def _reason_code(result: subprocess.CompletedProcess[str]) -> str:
        detail = (result.stderr or "").lower()
        if "no module named" in detail or "not found" in detail:
            return "dependency_missing"
        if "backend" in detail:
            return "backend_import_failed"
        return "probe_failed"

    def _run_worker(
        self,
        backend: ExecutionBackend,
        arguments: Sequence[str],
        *,
        input_text: str | None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["KERAS_BACKEND"] = keras_backend_value(backend)
        return subprocess.run(
            [sys.executable, "-m", "models.keras_worker", *arguments],
            cwd=self._server_root,
            env=environment,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
