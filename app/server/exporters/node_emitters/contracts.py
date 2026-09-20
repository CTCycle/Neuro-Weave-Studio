from __future__ import annotations

import keyword
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from common.catalogs.loader import NodeCatalog
from domain.compilation import CompiledNode
from domain.graph import ModelGraph
from domain.manifests import ImplementationMapping, NodeManifest, PortDefinition


###############################################################################
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
    anchor_expression: str | None = None


@dataclass(frozen=True)
class NodeRuntimeContext:
    """Normalized runtime inputs and backend services for one adapter call."""

    node_id: str
    inputs: tuple[Any, ...]
    source: Any | None
    normalized_parameters: Mapping[str, Any]
    runtime_services: Mapping[str, Any] = field(default_factory=dict)


###############################################################################
@dataclass(frozen=True)
class EmissionResult:
    setup_statements: tuple[str, ...] = ()
    output_expressions: Mapping[str, str] = field(default_factory=dict)


Emitter = Callable[[NodeEmitterContext], EmissionResult]
RuntimeEmitter = Callable[[NodeRuntimeContext], Any]


@dataclass
class _EmitterRegistration:
    emitter: Emitter
    runtime: RuntimeEmitter | None = None
    parameter_ids: frozenset[str] = frozenset()


###############################################################################
class NodeEmitterRegistry:
    """Backend-local adapter registry with fail-fast registration checks."""

    # -------------------------------------------------------------------------
    def __init__(self, backend_key: str) -> None:
        self.backend_key = backend_key
        self._emitters: dict[str, _EmitterRegistration] = {}

    # -------------------------------------------------------------------------
    def register(
        self,
        adapter_id: str,
        emitter: Emitter,
        *,
        runtime: RuntimeEmitter | None = None,
        parameter_ids: Collection[str] = (),
    ) -> None:
        if not adapter_id or not re.fullmatch(r"[a-z][a-z0-9_]*", adapter_id):
            raise ValueError(f"Invalid node emitter adapter id: {adapter_id!r}")
        if adapter_id in self._emitters:
            raise ValueError(
                f"Duplicate node emitter adapter '{adapter_id}' for {self.backend_key}"
            )
        self._emitters[adapter_id] = _EmitterRegistration(
            emitter=emitter,
            runtime=runtime,
            parameter_ids=frozenset(parameter_ids),
        )

    # -------------------------------------------------------------------------
    def declare_parameter_ids(self, adapter_id: str, parameter_ids: Collection[str]) -> None:
        try:
            registration = self._emitters[adapter_id]
        except KeyError as exc:
            raise KeyError(
                f"Cannot declare parameters for unknown node emitter '{adapter_id}'"
            ) from exc
        registration.parameter_ids = frozenset(parameter_ids)

    # -------------------------------------------------------------------------
    def resolve(self, adapter_id: str | None) -> Emitter:
        if not adapter_id or adapter_id not in self._emitters:
            raise KeyError(
                f"No node emitter '{adapter_id}' is registered for {self.backend_key}"
            )
        return self._emitters[adapter_id].emitter

    # -------------------------------------------------------------------------
    def has(self, adapter_id: str | None) -> bool:
        return bool(adapter_id and adapter_id in self._emitters)

    # -------------------------------------------------------------------------
    def resolve_runtime(self, adapter_id: str | None) -> RuntimeEmitter:
        if not adapter_id or adapter_id not in self._emitters:
            raise KeyError(
                f"No node runtime adapter '{adapter_id}' is registered for {self.backend_key}"
            )
        runtime = self._emitters[adapter_id].runtime
        if runtime is None:
            raise KeyError(
                f"Node emitter '{adapter_id}' has no runtime adapter for {self.backend_key}"
            )
        return runtime

    # -------------------------------------------------------------------------
    def has_runtime(self, adapter_id: str | None) -> bool:
        return bool(
            adapter_id
            and adapter_id in self._emitters
            and self._emitters[adapter_id].runtime is not None
        )

    # -------------------------------------------------------------------------
    def parameter_ids(self, adapter_id: str) -> frozenset[str]:
        try:
            return self._emitters[adapter_id].parameter_ids
        except KeyError as exc:
            raise KeyError(
                f"No node emitter '{adapter_id}' is registered for {self.backend_key}"
            ) from exc

    # -------------------------------------------------------------------------
    def validate_catalog(
        self,
        catalog: NodeCatalog,
        *,
        adapter_ids: Collection[str] | None = None,
    ) -> None:
        """Reject verified mappings that cannot be emitted or lose parameters."""

        errors: list[str] = []
        selected_adapters = set(adapter_ids) if adapter_ids is not None else None
        for manifest in catalog.manifests:
            if manifest.deprecated:
                continue
            mapping = manifest.implementations.get(self.backend_key)
            if mapping is None or not mapping.supported or mapping.status != "verified":
                continue
            if manifest.operation is not None and mapping.symbol is not None:
                continue
            if selected_adapters is not None and mapping.adapter not in selected_adapters:
                continue
            if not self.has(mapping.adapter):
                errors.append(
                    f"{manifest.id}: verified {self.backend_key} mapping requires adapter "
                    f"{mapping.adapter!r}"
                )
                continue
            consumed = self.parameter_ids(mapping.adapter or "")
            ignored = sorted(
                parameter.id
                for parameter in manifest.parameters
                if parameter.scope == "authoring"
                and parameter.id not in consumed
                and parameter.id not in mapping.constructor_omissions
            )
            if ignored:
                errors.append(
                    f"{manifest.id}: adapter {mapping.adapter!r} ignores editable parameters "
                    f"{', '.join(ignored)}"
                )
        if errors:
            raise ValueError("\n".join(errors))


###############################################################################
def safe_identifier(value: str, fallback: str = "value") -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        candidate = f"{fallback}_{candidate}"
    return candidate


###############################################################################
def render_structured_value(value: Any) -> str:
    """Render a JSON-safe parameter without introducing executable code."""

    return repr(value)


###############################################################################
def layout_permutation(rank: int, *, to_channels_first: bool) -> tuple[int, ...]:
    if rank < 3:
        return tuple(range(rank))
    if to_channels_first:
        return (0, rank - 1, *range(1, rank - 1))
    return (0, *range(2, rank), 1)


###############################################################################
def emit_multi_output_assignment(
    expression: str, output_ports: tuple[PortDefinition, ...]
) -> dict[str, str]:
    if len(output_ports) == 1:
        return {output_ports[0].id: expression}
    return {port.id: f"{expression}[{index}]" for index, port in enumerate(output_ports)}
