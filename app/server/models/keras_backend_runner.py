from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from domain.enums import ExecutionBackend, keras_backend_value
from domain.graph import ModelGraph

LOGGER = logging.getLogger(__name__)

###############################################################################
class KerasBackendRunner:
    """Probe and execute Keras in isolated workers with explicit provenance."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._server_root = Path(__file__).resolve().parents[1]

    # -------------------------------------------------------------------------
    def available_backends(self) -> tuple[ExecutionBackend, ...]:
        return tuple(
            backend for backend, status in self.probe_statuses().items() if status["available"]
        )

    # -------------------------------------------------------------------------
    def probe_statuses(self) -> dict[ExecutionBackend, dict[str, object]]:
        statuses: dict[ExecutionBackend, dict[str, object]] = {}
        for backend in ExecutionBackend:
            started = time.perf_counter()
            try:
                result = self._run_worker(backend, ("--probe",), input_text=None)
                capabilities = self._parse_probe_output(result)
                expected = keras_backend_value(backend)
                available = bool(
                    result.returncode == 0
                    and capabilities.get("backend") == expected
                    and capabilities.get("available", True)
                )
                reason = None if available else str(capabilities.get("reason_code") or self._reason_code(result))
            except (OSError, subprocess.SubprocessError):
                available = False
                reason = "worker_failed"
                capabilities = {}
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            statuses[backend] = {
                "available": available,
                "reason_code": reason,
                "accelerator": capabilities.get("accelerator") if available else None,
                "device_name": capabilities.get("device_name") if available else None,
            }
            LOGGER.info(
                "backend_probe requested=%s worker_backend=%s duration_ms=%s outcome=%s",
                backend.value,
                keras_backend_value(backend),
                duration_ms,
                "available" if available else reason,
            )
        return statuses

    # -------------------------------------------------------------------------
    def probe(self, backend: ExecutionBackend) -> bool:
        return bool(self.probe_statuses()[backend]["available"])

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def smoke_test(self, graph: ModelGraph, backend: ExecutionBackend) -> dict[str, object]:
        started = time.perf_counter()
        result = self._run_worker(
            backend,
            ("--smoke-test",),
            input_text=graph.model_dump_json(),
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(detail or "Keras inference worker failed without a diagnostic")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Keras inference worker returned invalid metadata") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Keras inference worker did not report a successful smoke test")
        payload = cast(dict[str, object], payload)
        if payload.get("status") != "succeeded":
            raise RuntimeError("Keras inference worker did not report a successful smoke test")
        payload["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        LOGGER.info(
            "backend_execution requested=%s worker_backend=%s duration_ms=%s outcome=smoke_test",
            backend.value,
            keras_backend_value(backend),
            payload["duration_ms"],
        )
        return payload

    # -------------------------------------------------------------------------
    @staticmethod
    def _reason_code(result: subprocess.CompletedProcess[str]) -> str:
        detail = (result.stderr or "").lower()
        if "no module named" in detail or "not found" in detail:
            return "dependency_missing"
        if "backend" in detail:
            return "backend_import_failed"
        return "probe_failed"

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_probe_output(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"backend": result.stdout.strip()}
        return cast(dict[str, object], payload) if isinstance(payload, dict) else {}

    # -------------------------------------------------------------------------
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
