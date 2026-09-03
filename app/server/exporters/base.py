from __future__ import annotations

from abc import ABC, abstractmethod

from common.catalogs.loader import NodeCatalog
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import CompilationCatalog
from domain.graph import ModelGraph


###############################################################################
class SourceExporter(ABC):
    # -------------------------------------------------------------------------
    @abstractmethod
    def generate(
        self,
        graph: ModelGraph,
        ir: CanonicalModelIR,
        catalog: NodeCatalog | None = None,
        *,
        compilation_catalog: CompilationCatalog | None = None,
    ) -> str:
        raise NotImplementedError
