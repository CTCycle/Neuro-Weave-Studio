from __future__ import annotations

import importlib.util
from uuid import UUID

from common.timestamps import utc_now
from domain.compilation import CompilationResult
from domain.enums import ExecutionBackend
from domain.exports import (
    SUPPORTED_ONNX_OPSETS,
    ExportCapabilitiesResponse,
    ExportCapability,
    ExportCategory,
    ExportConversion,
    ExportFormat,
    ExportLayout,
    ExportOptionDefinition,
    ExportOptions,
    ExportPayload,
    SourceExportResponse,
    export_suffix,
)
from domain.graph import ProjectDocument
from domain.runtime import RuntimeApi
from repositories.contracts import UnitOfWorkFactory
from services.compilation_service import CompilationService
from services.errors import (
    CompilationRejectedError,
    ProjectNotFoundError,
    RevisionConflictError,
)
from services.project_service import ProjectService
from services.runtime_controller import RuntimeController
from services.validation_service import ValidationService

###############################################################################
class ExportService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        project_service: ProjectService,
        compilation_service: CompilationService,
        uow_factory: UnitOfWorkFactory,
        runtime_controller: RuntimeController,
        validation_service: ValidationService,
    ) -> None:
        self._projects = project_service
        self._compilation = compilation_service
        self._uow_factory = uow_factory
        self._runtime = runtime_controller
        self._validator = validation_service

    # -------------------------------------------------------------------------
    def source(self, project_id: UUID) -> SourceExportResponse:
        project = self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        result: CompilationResult = self._compilation.compile(
            project.graph, project.execution_backend, project.keras_enabled
        )
        if not result.valid or result.code is None:
            raise CompilationRejectedError(result.diagnostics)

        with self._uow_factory() as uow:
            current = uow.projects.get_metadata(project_id)
            if current is None:
                raise ProjectNotFoundError(project_id)
            if current.current_revision != project.current_revision:
                raise RevisionConflictError(
                    "The project changed while source code was being generated"
                )
            uow.code_snapshots.upsert(
                project_id,
                revision=current.current_revision,
                execution_backend=current.execution_backend,
                keras_enabled=current.keras_enabled,
                source_code=result.code,
                timestamp=utc_now(),
            )
            uow.commit()
        return SourceExportResponse(
            execution_backend=project.execution_backend,
            keras_enabled=project.keras_enabled,
            code=result.code,
        )

    # -------------------------------------------------------------------------
    def capabilities(self, project_id: UUID) -> ExportCapabilitiesResponse:
        project = self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        validation = self._validator.validate(
            project.graph,
            project.execution_backend,
            project.keras_enabled,
        )
        diagnostics = [item.model_dump(mode="json") for item in validation.diagnostics]
        formats = self._format_capabilities(project, validation.valid)
        return ExportCapabilitiesResponse(
            project_id=project.id,
            project_revision=project.current_revision,
            execution_backend=project.execution_backend,
            keras_enabled=project.keras_enabled,
            valid=validation.valid,
            diagnostics=diagnostics,
            formats=formats,
        )

    # -------------------------------------------------------------------------
    def _format_capabilities(self, project: ProjectDocument, valid: bool) -> list[ExportCapability]:
        execution_backend = project.execution_backend  # type: ignore[attr-defined]
        keras_enabled = project.keras_enabled  # type: ignore[attr-defined]
        formats: list[ExportCapability] = []
        profile_api = RuntimeApi.KERAS if keras_enabled else RuntimeApi.NATIVE
        profile_api_status = self._runtime.api_status(execution_backend, profile_api)
        profile_available = bool(profile_api_status and profile_api_status.available)

        def add(
            export_format: ExportFormat,
            *,
            label: str,
            category: ExportCategory,
            conversion: ExportConversion,
            layout: ExportLayout,
            payload: ExportPayload,
            api: RuntimeApi | None = profile_api,
            backend: ExecutionBackend | None = execution_backend,
            available: bool = valid,
            reason_code: str | None = None,
            reason: str | None = None,
            limitations: list[str] | None = None,
            dependencies: list[str] | None = None,
            options: list[ExportOptionDefinition] | None = None,
            compatibility: str = "Current project revision",
        ) -> None:
            formats.append(
                ExportCapability(
                    format=export_format,
                    label=label,
                    category=category,
                    available=available,
                    reason_code=reason_code,
                    reason=reason,
                    execution_backend=backend,
                    runtime_api=api,
                    conversion=conversion,
                    layout=layout,
                    extension=export_suffix(export_format, ExportOptions()),
                    payload=payload,
                    compatibility=compatibility,
                    limitations=limitations
                    or ["Initialized inference artifact; training state is unavailable."],
                    dependencies=dependencies or [],
                    options=[_filename_option(), *(options or [])],
                )
            )

        add(
            ExportFormat.PROJECT_BUNDLE,
            label="Project bundle",
            category=ExportCategory.PROJECT,
            conversion=ExportConversion.CANONICAL,
            layout=ExportLayout.FILE,
            payload=ExportPayload.PROJECT,
            api=None,
            available=valid,
            reason_code=None if valid else "model_invalid",
            reason=None
            if valid
            else "Fix the validation diagnostics before exporting the project bundle.",
            compatibility="Canonical NeuroWeave graph, configuration, and execution profile",
        )
        add(
            ExportFormat.SOURCE,
            label="Generated source",
            category=ExportCategory.SOURCE,
            conversion=ExportConversion.CANONICAL,
            layout=ExportLayout.FILE,
            payload=ExportPayload.SOURCE,
            api=profile_api,
            available=valid,
            reason_code=None if valid else "source_unavailable",
            reason=None if valid else "The current graph does not compile for its saved profile.",
            compatibility="Generated source for the saved execution profile",
        )

        keras_available = valid and keras_enabled and profile_available
        keras_reason_code, keras_reason = _runtime_reason(profile_api_status, "Keras runtime")
        add(
            ExportFormat.KERAS,
            label="Keras model",
            category=ExportCategory.NATIVE,
            conversion=ExportConversion.KERAS,
            layout=ExportLayout.FILE,
            payload=ExportPayload.FULL_MODEL,
            available=keras_available,
            reason_code=None
            if keras_available
            else "keras_disabled"
            if not keras_enabled
            else keras_reason_code,
            reason=None
            if keras_available
            else "Keras is disabled for this native profile."
            if not keras_enabled
            else keras_reason,
            limitations=[
                "Inference-only initialized model; optimizer and training state are not preserved."
            ],
            dependencies=["keras", _backend_package(execution_backend)],
            compatibility="Keras-enabled profiles with an operational selected backend",
        )
        add(
            ExportFormat.KERAS_WEIGHTS,
            label="Keras weights",
            category=ExportCategory.CHECKPOINT,
            conversion=ExportConversion.KERAS,
            layout=ExportLayout.FILE,
            payload=ExportPayload.WEIGHTS_ONLY,
            available=keras_available,
            reason_code=None
            if keras_available
            else "keras_disabled"
            if not keras_enabled
            else keras_reason_code,
            reason=None
            if keras_available
            else "Weights-only Keras export requires the Keras runtime API."
            if not keras_enabled
            else keras_reason,
            dependencies=["keras", _backend_package(execution_backend)],
            compatibility="Keras-enabled profiles",
        )

        tf_saved_available = (
            valid and execution_backend is ExecutionBackend.TENSORFLOW and profile_available
        )
        add(
            ExportFormat.TENSORFLOW_SAVED_MODEL,
            label="TensorFlow SavedModel",
            category=ExportCategory.NATIVE,
            conversion=ExportConversion.KERAS if keras_enabled else ExportConversion.NATIVE,
            layout=ExportLayout.ARCHIVE,
            payload=ExportPayload.FULL_MODEL,
            available=tf_saved_available,
            reason_code=None
            if tf_saved_available
            else "backend_unsupported"
            if execution_backend is not ExecutionBackend.TENSORFLOW
            else keras_reason_code,
            reason=None
            if tf_saved_available
            else "SavedModel export requires the TensorFlow execution backend and runtime API."
            if execution_backend is not ExecutionBackend.TENSORFLOW
            else keras_reason,
            dependencies=["tensorflow"],
            compatibility="Keras+TensorFlow or native TensorFlow profiles",
        )

        onnx_dependencies = ["onnx", "onnxruntime", "onnxscript"]
        if execution_backend is ExecutionBackend.JAX:
            onnx_dependencies.append("jax2onnx")
        elif execution_backend is not ExecutionBackend.PYTORCH:
            onnx_dependencies.append("tf2onnx")
        onnx_missing = _missing_packages(onnx_dependencies)
        onnx_available = valid and profile_available and not onnx_missing
        onnx_reason_code = (
            "dependency_missing"
            if onnx_missing
            else keras_reason_code
        )
        onnx_reason = (
            f"Install {', '.join(onnx_missing)} to enable verified ONNX export."
            if onnx_missing
            else keras_reason
        )
        add(
            ExportFormat.ONNX,
            label="ONNX interchange",
            category=ExportCategory.INTERCHANGE,
            conversion=ExportConversion.KERAS
            if keras_enabled
            else ExportConversion.NATIVE
            if execution_backend in {ExecutionBackend.PYTORCH, ExecutionBackend.JAX}
            else ExportConversion.BRIDGE,
            layout=ExportLayout.FILE,
            payload=ExportPayload.FULL_MODEL,
            available=onnx_available,
            reason_code=None if onnx_available else onnx_reason_code,
            reason=None if onnx_available else onnx_reason,
            dependencies=onnx_dependencies,
            options=[
                ExportOptionDefinition(
                    id="opset_version",
                    label="ONNX opset",
                    type="integer",
                    default=18,
                    choices=list(SUPPORTED_ONNX_OPSETS),
                ),
                ExportOptionDefinition(
                    id="dynamic_batch", label="Dynamic batch", type="boolean", default=True
                ),
                ExportOptionDefinition(
                    id="external_data",
                    label="External data",
                    type="choice",
                    default="auto",
                    choices=["auto", "always", "never"],
                ),
            ],
            compatibility="Verified inference interchange with named inputs/outputs and optional external data",
        )

        pytorch_native = (
            valid
            and execution_backend is ExecutionBackend.PYTORCH
            and not keras_enabled
            and profile_available
        )
        native_reason_code, native_reason = _runtime_reason(
            self._runtime.api_status(execution_backend, RuntimeApi.NATIVE),
            "Native runtime",
        )
        add(
            ExportFormat.PYTORCH_EXPORTED_PROGRAM,
            label="PyTorch ExportedProgram",
            category=ExportCategory.NATIVE,
            conversion=ExportConversion.NATIVE,
            layout=ExportLayout.FILE,
            payload=ExportPayload.FULL_MODEL,
            api=RuntimeApi.NATIVE,
            available=pytorch_native,
            reason_code=None
            if pytorch_native
            else "backend_unsupported"
            if execution_backend is not ExecutionBackend.PYTORCH
            else "keras_enabled"
            if keras_enabled
            else native_reason_code,
            reason=None
            if pytorch_native
            else "This artifact is available only for native PyTorch profiles."
            if execution_backend is not ExecutionBackend.PYTORCH or keras_enabled
            else native_reason,
            dependencies=["torch"],
            compatibility="Native PyTorch inference program without optimizer state",
        )

        safe_dependencies = ["safetensors"]
        safe_missing = _missing_packages(safe_dependencies)
        safe_available = valid and profile_available and not safe_missing
        add(
            ExportFormat.SAFETENSORS,
            label="SafeTensors package",
            category=ExportCategory.CHECKPOINT,
            conversion=ExportConversion.KERAS if keras_enabled else ExportConversion.NATIVE,
            layout=ExportLayout.ARCHIVE,
            payload=ExportPayload.WEIGHTS_ONLY,
            available=safe_available,
            reason_code=None
            if safe_available
            else "dependency_missing"
            if safe_missing
            else keras_reason_code,
            reason=None
            if safe_available
            else f"Install {', '.join(safe_missing)} to enable verified SafeTensors extraction."
            if safe_missing
            else keras_reason,
            dependencies=safe_dependencies,
            compatibility="Weights-only package with the canonical graph/profile manifest; not universally loadable",
        )

        checkpoint_available = (
            valid
            and execution_backend is ExecutionBackend.TENSORFLOW
            and not keras_enabled
            and profile_available
        )
        add(
            ExportFormat.TENSORFLOW_CHECKPOINT,
            label="TensorFlow checkpoint",
            category=ExportCategory.CHECKPOINT,
            conversion=ExportConversion.NATIVE,
            layout=ExportLayout.ARCHIVE,
            payload=ExportPayload.WEIGHTS_ONLY,
            api=RuntimeApi.NATIVE,
            available=checkpoint_available,
            reason_code=None
            if checkpoint_available
            else "backend_unsupported"
            if execution_backend is not ExecutionBackend.TENSORFLOW or keras_enabled
            else native_reason_code,
            reason=None
            if checkpoint_available
            else "Weights-only TensorFlow checkpoints require a native TensorFlow profile."
            if execution_backend is not ExecutionBackend.TENSORFLOW or keras_enabled
            else native_reason,
            dependencies=["tensorflow"],
            compatibility="Native TensorFlow weights plus graph/profile manifest",
        )
        flax_available = (
            valid
            and execution_backend is ExecutionBackend.JAX
            and not keras_enabled
            and profile_available
        )
        add(
            ExportFormat.FLAX_CHECKPOINT,
            label="Flax checkpoint",
            category=ExportCategory.CHECKPOINT,
            conversion=ExportConversion.NATIVE,
            layout=ExportLayout.ARCHIVE,
            payload=ExportPayload.WEIGHTS_ONLY,
            api=RuntimeApi.NATIVE,
            available=flax_available,
            reason_code=None
            if flax_available
            else "backend_unsupported"
            if execution_backend is not ExecutionBackend.JAX or keras_enabled
            else native_reason_code,
            reason=None
            if flax_available
            else "Weights-only Flax checkpoints require a native JAX profile."
            if execution_backend is not ExecutionBackend.JAX or keras_enabled
            else native_reason,
            dependencies=["jax", "flax"],
            compatibility="Native JAX parameter/message-pack data plus graph/profile manifest",
        )
        return formats

###############################################################################
def _filename_option() -> ExportOptionDefinition:
    return ExportOptionDefinition(
        id="filename_stem",
        label="Filename stem",
        type="string",
        default=None,
        description="Server sanitizes the stem and derives the extension from the selected format.",
    )

###############################################################################
def _backend_package(backend: ExecutionBackend) -> str:
    return {
        ExecutionBackend.PYTORCH: "torch",
        ExecutionBackend.TENSORFLOW: "tensorflow",
        ExecutionBackend.JAX: "jax",
    }[backend]

###############################################################################
def _missing_packages(packages: list[str]) -> list[str]:
    missing: list[str] = []
    for package in packages:
        module = {
            "onnxruntime": "onnxruntime",
            "onnxscript": "onnxscript",
            "tf2onnx": "tf2onnx",
        }.get(package, package)
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            missing.append(package)
    return missing

###############################################################################
def _runtime_reason(status: object, label: str) -> tuple[str, str]:
    if status is None:
        return "runtime_unavailable", f"{label} capability has not been probed."
    if getattr(status, "available", False):
        return "", ""
    reason_code = str(getattr(status, "reason_code", None) or "runtime_unavailable")
    return reason_code, f"{label} is unavailable ({reason_code.replace('_', ' ')})."
