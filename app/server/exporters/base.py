from __future__ import annotations

from abc import ABC, abstractmethod

from common.catalogs.loader import NodeCatalog
from domain.compilation import CanonicalModelIR
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
    ) -> str:
        raise NotImplementedError
