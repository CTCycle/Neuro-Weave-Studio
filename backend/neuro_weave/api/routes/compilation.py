from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from neuro_weave.api.dependencies import get_compilation_service
from neuro_weave.domain.compilation import CompilationResult
from neuro_weave.domain.enums import TargetFramework
from neuro_weave.domain.graph import ModelGraph
from neuro_weave.services.compilation_service import CompilationService

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
