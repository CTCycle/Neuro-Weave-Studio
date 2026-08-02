from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_validator
from domain.diagnostics import ValidationResult
from domain.enums import TargetFramework
from domain.graph import ModelGraph
from validation.validator import GraphValidator

router = APIRouter(prefix="/api/validation", tags=["validation"])


class ValidateRequest(BaseModel):
    graph: ModelGraph
    framework: TargetFramework


@router.post("", response_model=ValidationResult)
def validate_graph(
    payload: ValidateRequest,
    validator: GraphValidator = Depends(get_validator),
) -> ValidationResult:
    return validator.validate(payload.graph, payload.framework)
