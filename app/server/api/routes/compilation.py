from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_compilation_service
from domain.compilation import CompilationResult
from domain.enums import TargetFramework
from domain.graph import ModelGraph
from services.compilation_service import CompilationService

router = APIRouter(prefix="/api/compilation", tags=["compilation"])


class CompileRequest(BaseModel):
    graph: ModelGraph
    framework: TargetFramework


@router.post("", response_model=CompilationResult)
def compile_graph(
    payload: CompileRequest,
    service: CompilationService = Depends(get_compilation_service),
) -> CompilationResult:
    return service.compile(payload.graph, payload.framework)
