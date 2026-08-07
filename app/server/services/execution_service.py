from __future__ import annotations

import subprocess
from collections.abc import Mapping
from typing import cast

from common.catalogs.loader import NodeCatalog
from domain.diagnostics import ValidationDiagnostic
from domain.enums import CodeTarget, ExecutionBackend
from domain.execution import SmokeTestResponse, TensorExecutionInfo
from domain.graph import ModelGraph
from models.keras_backend_runner import KerasBackendRunner
from services.validation_service import ValidationService


###############################################################################
class SmokeTestValidationError(ValueError):
    def __init__(self, diagnostics: list[ValidationDiagnostic]) -> None:
        super().__init__("Model graph is not ready for an inference smoke test")
        self.diagnostics = diagnostics


###############################################################################
class SmokeTestBackendUnavailableError(RuntimeError):
    def __init__(self, backend: ExecutionBackend, reason_code: str | None) -> None:
        self.backend = backend.value
        self.reason_code = reason_code or "backend_unavailable"
        super().__init__(f"Execution backend '{self.backend}' is not operational")


###############################################################################
class SmokeTestExecutionError(RuntimeError):
    def __init__(self, message: str, *, code: str = "execution_failed") -> None:
        self.code = code
        super().__init__(message)


###############################################################################
class ExecutionService:
    """Coordinates validation and one isolated, inference-only Keras worker run."""

    def __init__(
        self,
        catalog: NodeCatalog,
        backend_runner: KerasBackendRunner,
        backend_statuses: Mapping[ExecutionBackend, Mapping[str, object]],
    ) -> None:
        self._validator = ValidationService(catalog)
        self._backend_runner = backend_runner
        self._backend_statuses = backend_statuses

    def smoke_test(
        self, graph: ModelGraph, execution_backend: ExecutionBackend
    ) -> SmokeTestResponse:
        validation = self._validator.validate(graph, CodeTarget.KERAS)
        if not validation.valid:
            raise SmokeTestValidationError(validation.diagnostics)

        status = self._backend_statuses.get(execution_backend)
        if status is None or not bool(status.get("available")):
            reason = str(status.get("reason_code")) if status and status.get("reason_code") else None
            raise SmokeTestBackendUnavailableError(execution_backend, reason)

        try:
            result = self._backend_runner.smoke_test(graph, execution_backend)
        except RuntimeError as exc:
            message = str(exc) or "The inference worker failed without a diagnostic"
            code = "unsupported_layer" if "does not support node" in message else "execution_failed"
            raise SmokeTestExecutionError(message, code=code) from exc
        except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
            raise SmokeTestExecutionError(str(exc) or "The inference worker could not start", code="worker_failed") from exc

        inputs = cast(list[dict[str, object]], result.get("inputs", []))
        outputs = cast(list[dict[str, object]], result.get("outputs", []))
        parameter_count = cast(int, result.get("parameter_count", 0))
        duration_ms = cast(float, result.get("duration_ms", 0.0))
        return SmokeTestResponse(
            execution_backend=execution_backend,
            accelerator=str(result.get("accelerator") or "cpu"),
            device_name=str(result.get("device_name") or "CPU"),
            inputs=[TensorExecutionInfo.model_validate(item) for item in inputs],
            outputs=[TensorExecutionInfo.model_validate(item) for item in outputs],
            parameter_count=int(parameter_count),
            duration_ms=float(duration_ms),
        )
