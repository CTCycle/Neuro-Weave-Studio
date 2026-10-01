# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pydantic import BaseModel

from domain.enums import ExecutionBackend


###############################################################################
class ModelExecutionProfile(BaseModel):
    """The persisted architecture choice for one NeuroWeave model."""

    execution_backend: ExecutionBackend
    keras_enabled: bool = True

    # -------------------------------------------------------------------------
    @property
    def implementation_key(self) -> str:
        return "keras" if self.keras_enabled else self.execution_backend.value
