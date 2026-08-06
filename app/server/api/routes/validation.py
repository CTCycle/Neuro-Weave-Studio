from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_validation_service
from domain.diagnostics import ValidationRequest, ValidationResult
from services.validation_service import ValidationService

router = APIRouter(prefix="/api/validation", tags=["validation"])

###############################################################################
@router.post("", response_model=ValidationResult)
def validate_graph(
    payload: ValidationRequest,
    service: ValidationService = Depends(get_validation_service),
) -> ValidationResult:
    return service.validate(payload.graph, payload.code_target)
