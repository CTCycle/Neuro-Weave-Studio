from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_runtime_service
from domain.metadata import HealthResponse
from services.runtime_service import RuntimeService

router = APIRouter(tags=["health"])


###############################################################################
@router.get("/health", response_model=HealthResponse)
def health(service: RuntimeService = Depends(get_runtime_service)) -> HealthResponse:
    return service.health()
