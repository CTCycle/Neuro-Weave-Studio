from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from neuro_weave.api.dependencies import get_validator
from neuro_weave.domain.diagnostics import ValidationResult
from neuro_weave.domain.enums import TargetFramework
from neuro_weave.domain.graph import ModelGraph
from neuro_weave.validation.validator import GraphValidator

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
