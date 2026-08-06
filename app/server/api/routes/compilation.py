from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_compilation_service
from domain.compilation import CompilationRequest, CompilationResult
from services.compilation_service import CompilationService

router = APIRouter(prefix="/api/compilation", tags=["compilation"])

###############################################################################
@router.post("", response_model=CompilationResult)
def compile_graph(
    payload: CompilationRequest,
    service: CompilationService = Depends(get_compilation_service),
) -> CompilationResult:
    return service.compile(payload.graph, payload.code_target)
