from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from domain.manifests import NodeManifest

RESOURCE_ROOT = Path(__file__).resolve().parents[2] / "resources" / "nodes"


class NodeCatalog:
    def __init__(self, manifests: list[NodeManifest]) -> None:
        self._manifests = sorted(manifests, key=lambda item: (item.category, item.display_name))
        self._by_key = {(item.id, item.version): item for item in self._manifests}

    @property
    def manifests(self) -> list[NodeManifest]:
        return list(self._manifests)

    def get(self, definition_id: str, version: int = 1) -> NodeManifest:
        try:
            return self._by_key[(definition_id, version)]
        except KeyError as exc:
            raise KeyError(f"Unknown node definition: {definition_id}@{version}") from exc


def get_node_catalog() -> NodeCatalog:
    manifests: list[NodeManifest] = []
    for path in sorted(RESOURCE_ROOT.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = cast(
            list[dict[str, Any]], raw if isinstance(raw, list) else raw["nodes"]
        )
        manifests.extend(NodeManifest.model_validate(item) for item in items)
    if not manifests:
        raise RuntimeError(f"No node manifests found under {RESOURCE_ROOT}")
    keys = [(manifest.id, manifest.version) for manifest in manifests]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate node manifest identity detected")
    return NodeCatalog(manifests)
