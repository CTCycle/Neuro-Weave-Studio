from __future__ import annotations

from pydantic import BaseModel

from domain.enums import ExecutionBackend

###############################################################################
class SourceExportResponse(BaseModel):
    execution_backend: ExecutionBackend
    keras_enabled: bool
    code: str

###############################################################################
