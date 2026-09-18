from __future__ import annotations

import keyword
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from common.catalogs.loader import NodeCatalog
from domain.compilation import CompiledNode
from domain.graph import ModelGraph
from domain.manifests import ImplementationMapping, NodeManifest, PortDefinition


@dataclass(frozen=True)
class NodeEmitterContext:
    """The complete normalized contract available to one backend emitter."""

    graph: ModelGraph
    canonical_ir_node: CompiledNode
    manifest: NodeManifest
    implementation_mapping: ImplementationMapping
    normalized_parameters: Mapping[str, Any]
    input_expressions_by_target_port: Mapping[str, str]
    materialized_output_ports: tuple[PortDefinition, ...]


@dataclass(frozen=True)
class EmissionResult:
    setup_statements: tuple[str, ...] = ()
    output_expressions: Mapping[str, str] = field(default_factory=dict)


Emitter = Callable[[NodeEmitterContext], EmissionResult]


class NodeEmitterRegistry:
    """Backend-local adapter registry with fail-fast registration checks."""

    def __init__(self, backend_key: str) -> None:
        self.backend_key = backend_key
        self._emitters: dict[str, Emitter] = {}

    def register(self, adapter_id: str, emitter: Emitter) -> None:
        if not adapter_id or not re.fullmatch(r"[a-z][a-z0-9_]*", adapter_id):
            raise ValueError(f"Invalid node emitter adapter id: {adapter_id!r}")
        if adapter_id in self._emitters:
            raise ValueError(
                f"Duplicate node emitter adapter '{adapter_id}' for {self.backend_key}"
            )
        self._emitters[adapter_id] = emitter

    def resolve(self, adapter_id: str | None) -> Emitter:
        if not adapter_id or adapter_id not in self._emitters:
            raise KeyError(
                f"No node emitter '{adapter_id}' is registered for {self.backend_key}"
            )
        return self._emitters[adapter_id]

    def has(self, adapter_id: str | None) -> bool:
        return bool(adapter_id and adapter_id in self._emitters)

    def validate_catalog(self, catalog: NodeCatalog) -> None:
        """Reject verified supported mappings that cannot be emitted."""

        errors: list[str] = []
        for manifest in catalog.manifests:
            if manifest.deprecated:
                continue
            mapping = manifest.implementations.get(self.backend_key)
            if mapping is None or not mapping.supported or mapping.status != "verified":
                continue
            if manifest.operation is not None and mapping.symbol is not None:
                continue
            if not self.has(mapping.adapter):
                errors.append(
                    f"{manifest.id}: verified {self.backend_key} mapping requires adapter "
                    f"{mapping.adapter!r}"
                )
        if errors:
            raise ValueError("\n".join(errors))


def safe_identifier(value: str, fallback: str = "value") -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        candidate = f"{fallback}_{candidate}"
    return candidate


def render_structured_value(value: Any) -> str:
    """Render a JSON-safe parameter without introducing executable code."""

    return repr(value)


def layout_permutation(rank: int, *, to_channels_first: bool) -> tuple[int, ...]:
    if rank < 3:
        return tuple(range(rank))
    if to_channels_first:
        return (0, rank - 1, *range(1, rank - 1))
    return (0, *range(2, rank), 1)


def emit_multi_output_assignment(
    expression: str, output_ports: tuple[PortDefinition, ...]
) -> dict[str, str]:
    if len(output_ports) == 1:
        return {output_ports[0].id: expression}
    return {port.id: f"{expression}[{index}]" for index, port in enumerate(output_ports)}
