from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_validation_service
from domain.diagnostics import (
    ConnectionValidationRequest,
    ConnectionValidationResult,
    ValidationRequest,
    ValidationResult,
)
from services.validation_service import ValidationService

router = APIRouter(prefix="/api/validation", tags=["validation"])

###############################################################################
@router.post("", response_model=ValidationResult)
def validate_graph(
    payload: ValidationRequest,
    service: ValidationService = Depends(get_validation_service),
) -> ValidationResult:
    return service.validate(payload.graph, payload.execution_backend, payload.keras_enabled)

###############################################################################
@router.post("/connection", response_model=ConnectionValidationResult)
def validate_connection(
    payload: ConnectionValidationRequest,
    service: ValidationService = Depends(get_validation_service),
) -> ConnectionValidationResult:
    return service.validate_connection(
        payload.graph,
        payload.connection,
        payload.execution_backend,
        payload.keras_enabled,
    )
