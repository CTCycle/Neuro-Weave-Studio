from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.dependencies import get_job_service, get_runtime_controller
from domain.enums import ExecutionBackend
from domain.jobs import JobResponse, JobSubmitRequest, TensorInspectionOperation
from domain.runtime import RuntimeApi, RuntimeDiscoveryRequest, RuntimeDiscoveryResponse
from services.errors import CompilationRejectedError, ProjectNotFoundError
from services.job_service import JobService
from services.runtime_controller import RuntimeController, RuntimeControllerError

router = APIRouter(prefix="/api/execution", tags=["execution"])

###############################################################################
@router.post("/discovery", response_model=RuntimeDiscoveryResponse, status_code=202)
def request_discovery(
    payload: RuntimeDiscoveryRequest,
    controller: RuntimeController = Depends(get_runtime_controller),
) -> RuntimeDiscoveryResponse:
    try:
        return controller.request_discovery(
            payload.backend,
            payload.api,
            force_refresh=payload.force_refresh,
        )
    except RuntimeControllerError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc), **exc.details},
        ) from exc

###############################################################################
@router.get("/discovery", response_model=RuntimeDiscoveryResponse)
def discovery_status(
    backend: str,
    api: str = "keras",
    controller: RuntimeController = Depends(get_runtime_controller),
) -> RuntimeDiscoveryResponse:
    try:
        return controller.discovery_status(ExecutionBackend(backend), RuntimeApi(api))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Unknown backend or runtime API") from exc

###############################################################################
@router.post("/jobs", response_model=JobResponse, status_code=202)
def submit_job(
    payload: JobSubmitRequest,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    try:
        return service.submit(payload)
    except CompilationRejectedError as exc:
        stable_code = "model_validation_failed"
        if isinstance(payload.operation, TensorInspectionOperation) and exc.diagnostics:
            stable_code = exc.diagnostics[0].code
        raise HTTPException(
            status_code=422,
            detail={
                "code": stable_code,
                "message": str(exc),
                "diagnostics": [item.model_dump(mode="json") for item in exc.diagnostics],
            },
        ) from exc
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail={"code": "project_not_found", "message": str(exc)}
        ) from exc
    except RuntimeControllerError as exc:
        stable_code = exc.code
        if isinstance(payload.operation, TensorInspectionOperation) and exc.code in {
            "api_unavailable",
            "backend_unavailable",
            "runtime_discovery_pending",
            "runtime_discovery_required",
            "device_unavailable",
            "native_compiler_unavailable",
        }:
            stable_code = "runtime_unavailable"
        status = (
            409
            if exc.code
            in {
                "api_unavailable",
                "backend_unavailable",
                "runtime_discovery_pending",
                "runtime_discovery_required",
                "device_unavailable",
                "device_busy",
                "controller_shutting_down",
            }
            else 422
        )
        raise HTTPException(
            status_code=status,
            detail={"code": stable_code, "message": str(exc), "reason_code": exc.code, **exc.details},
        ) from exc

###############################################################################
@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(
    active_only: bool = Query(default=False),
    controller: RuntimeController = Depends(get_runtime_controller),
) -> list[JobResponse]:
    return controller.list(active_only=active_only)

###############################################################################
@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    controller: RuntimeController = Depends(get_runtime_controller),
) -> JobResponse:
    try:
        return controller.get(job_id)
    except RuntimeControllerError as exc:
        raise HTTPException(
            status_code=404, detail={"code": exc.code, "message": str(exc)}
        ) from exc

###############################################################################
@router.post("/jobs/{job_id}/stop", response_model=JobResponse)
def stop_job(
    job_id: str,
    controller: RuntimeController = Depends(get_runtime_controller),
) -> JobResponse:
    try:
        return controller.stop(job_id)
    except RuntimeControllerError as exc:
        status = 404 if exc.code == "job_not_found" else 409
        raise HTTPException(
            status_code=status, detail={"code": exc.code, "message": str(exc)}
        ) from exc

###############################################################################
@router.get("/jobs/{job_id}/artifact")
def get_job_artifact(
    job_id: str,
    controller: RuntimeController = Depends(get_runtime_controller),
) -> FileResponse:
    try:
        path, filename = controller.artifact_path(job_id)
    except RuntimeControllerError as exc:
        status = 404 if exc.code in {"job_not_found", "artifact_not_found"} else 409
        raise HTTPException(
            status_code=status, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    return FileResponse(path=path, filename=filename, media_type="application/octet-stream")
