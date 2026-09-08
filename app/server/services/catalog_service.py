from __future__ import annotations

from common.catalogs.loader import NodeCatalog
from domain.manifests import NodeCatalogResponse


###############################################################################
class CatalogService:
    # -------------------------------------------------------------------------
    def __init__(self, catalog: NodeCatalog) -> None:
        self._catalog = catalog

    # -------------------------------------------------------------------------
    def list(self) -> NodeCatalogResponse:
        return NodeCatalogResponse(nodes=self._catalog.manifests)
