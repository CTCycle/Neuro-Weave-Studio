from __future__ import annotations

from pydantic import BaseModel

from domain.enums import KerasBackend, TargetFramework


class SourceExportResponse(BaseModel):
    framework: TargetFramework
    code: str


class KerasBinaryExportRequest(BaseModel):
    backend: KerasBackend
