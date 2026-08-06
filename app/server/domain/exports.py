from __future__ import annotations

from pydantic import BaseModel

from domain.enums import CodeTarget, ExecutionBackend

###############################################################################
class SourceExportResponse(BaseModel):
    code_target: CodeTarget
    code: str

###############################################################################
class KerasBinaryExportRequest(BaseModel):
    execution_backend: ExecutionBackend
