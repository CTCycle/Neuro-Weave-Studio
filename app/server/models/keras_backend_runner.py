from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from domain.enums import KerasBackend
from domain.graph import ModelGraph


class KerasBackendRunner:
    """Probe and execute Keras in an isolated process with an explicit backend."""

    def __init__(self) -> None:
        self._server_root = Path(__file__).resolve().parents[1]

    def available_backends(self) -> tuple[KerasBackend, ...]:
        return tuple(backend for backend in KerasBackend if self.probe(backend))

    def probe(self, backend: KerasBackend) -> bool:
        try:
            result = self._run_worker(backend, ("--probe",), input_text=None)
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and result.stdout.strip() == backend.value

    def export(self, graph: ModelGraph, destination: Path, backend: KerasBackend) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
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
        return destination

    def _run_worker(
        self,
        backend: KerasBackend,
        arguments: Sequence[str],
        *,
        input_text: str | None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["KERAS_BACKEND"] = backend.value
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
