# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from common.catalogs.loader import NodeCatalog

from .audit import build_catalog_adapter_audit
from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    NodeRuntimeContext,
    emit_multi_output_assignment,
    layout_permutation,
    render_structured_value,
    safe_identifier,
    safe_index_expression,
)
from .jax import build_registry as build_jax_registry
from .keras import build_registry as build_keras_registry
from .pytorch import build_registry as build_pytorch_registry
from .tensorflow import build_registry as build_tensorflow_registry

REGISTRIES = {
    "keras": build_keras_registry(),
    "tensorflow": build_tensorflow_registry(),
    "pytorch": build_pytorch_registry(),
    "jax": build_jax_registry(),
}


###############################################################################
def registry_for_backend(backend_key: str) -> NodeEmitterRegistry:
    try:
        return REGISTRIES[backend_key]
    except KeyError as exc:
        raise KeyError(f"Unknown node emitter backend: {backend_key}") from exc


###############################################################################
def validate_emitter_coverage(catalog: NodeCatalog) -> None:
    for registry in REGISTRIES.values():
        registry.validate_catalog(catalog)


__all__ = [
    "EmissionResult",
    "NodeEmitterContext",
    "NodeEmitterRegistry",
    "NodeRuntimeContext",
    "build_catalog_adapter_audit",
    "emit_multi_output_assignment",
    "layout_permutation",
    "registry_for_backend",
    "render_structured_value",
    "safe_identifier",
    "safe_index_expression",
    "validate_emitter_coverage",
]
