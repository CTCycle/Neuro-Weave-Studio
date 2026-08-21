from __future__ import annotations

from pydantic import BaseModel

from domain.graph import ProjectDocument

###############################################################################
class KerasArchitectureImportResponse(BaseModel):
    project: ProjectDocument
    warnings: list[str]
