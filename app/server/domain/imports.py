from __future__ import annotations

from pydantic import BaseModel, RootModel

from domain.graph import ProjectDocument


###############################################################################
class KerasArchitectureImportRequest(RootModel[bytes]):
    """Raw .keras ZIP bytes accepted by the architecture import endpoint."""


###############################################################################
class KerasArchitectureImportResponse(BaseModel):
    project: ProjectDocument
    warnings: list[str]
