from __future__ import annotations

from pydantic import BaseModel

from domain.enums import CodeTarget

###############################################################################
class SourceExportResponse(BaseModel):
    code_target: CodeTarget
    code: str

###############################################################################
