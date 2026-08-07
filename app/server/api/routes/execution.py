from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_execution_service
from domain.execution import SmokeTestRequest, SmokeTestResponse
from services.execution_service import (
    ExecutionService,
    SmokeTestBackendUnavailableError,
    SmokeTestExecutionError,
    SmokeTestValidationError,
)

router = APIRouter(prefix="/api/execution", tags=["execution"])

###############################################################################
@router.post("/smoke-test", response_model=SmokeTestResponse)
def smoke_test(
    payload: SmokeTestRequest,
    service: ExecutionService = Depends(get_execution_service),
) -> SmokeTestResponse:
    try:
        return service.smoke_test(payload.graph, payload.execution_backend)
    except SmokeTestValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_validation_failed",
                "message": str(exc),
                "diagnostics": [item.model_dump(mode="json") for item in exc.diagnostics],
            },
        ) from exc
    except SmokeTestBackendUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "backend_unavailable",
                "backend": exc.backend,
                "reason_code": exc.reason_code,
                "message": str(exc),
            },
        ) from exc
    except SmokeTestExecutionError as exc:
        status_code = 422 if exc.code in {"unsupported_layer", "invalid_input_shape"} else 500
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
