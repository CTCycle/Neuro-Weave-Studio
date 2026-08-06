from __future__ import annotations

from domain.diagnostics import ValidationDiagnostic


###############################################################################
class ProjectNotFoundError(LookupError):

    # -------------------------------------------------------------------------
    def __init__(self, project_id: object) -> None:
        super().__init__(f"Project not found: {project_id}")


###############################################################################
class RevisionConflictError(ValueError):
    pass


###############################################################################
class GraphPersistenceError(RuntimeError):
    pass


###############################################################################
class CompilationRejectedError(ValueError):

    # -------------------------------------------------------------------------
    def __init__(self, diagnostics: list[ValidationDiagnostic]) -> None:
        super().__init__("Graph cannot be compiled")
        self.diagnostics = diagnostics


###############################################################################
class BinaryExportUnavailableError(RuntimeError):
    def __init__(self, backend: object) -> None:
        self.backend = str(getattr(backend, "value", backend))
        super().__init__(f"Execution backend '{self.backend}' is not operational")
