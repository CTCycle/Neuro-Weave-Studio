from __future__ import annotations

from pathlib import Path
from typing import Any

from common.catalogs.loader import NodeCatalog

from .contracts import NodeEmitterRegistry

BACKENDS = ("keras", "tensorflow", "pytorch", "jax")
NATIVE_BACKENDS = ("tensorflow", "pytorch", "jax")


def _mapping_counts(catalog: NodeCatalog, backend: str) -> dict[str, int]:
    counts = {
        "verified": 0,
        "unsupported": 0,
        "unverified": 0,
        "missing": 0,
    }
    for manifest in catalog.manifests:
        if manifest.deprecated:
            continue
        mapping = manifest.implementations.get(backend)
        if mapping is None:
            counts["missing"] += 1
        elif not mapping.supported or mapping.status == "unsupported":
            counts["unsupported"] += 1
        elif mapping.status == "unverified":
            counts["unverified"] += 1
        elif mapping.status == "verified":
            counts["verified"] += 1
    return counts


def _adapter_records(
    catalog: NodeCatalog,
    backend: str,
    registry: NodeEmitterRegistry,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    unresolved: list[dict[str, str]] = []
    ignored: list[dict[str, Any]] = []
    for manifest in catalog.manifests:
        if manifest.deprecated:
            continue
        mapping = manifest.implementations.get(backend)
        if (
            mapping is None
            or not mapping.supported
            or mapping.status != "verified"
            or (manifest.operation is not None and mapping.symbol is not None)
        ):
            continue
        if mapping.adapter is None or not registry.has(mapping.adapter):
            unresolved.append(
                {
                    "manifest": manifest.id,
                    "adapter": mapping.adapter or "<none>",
                    "reason": "missing_registry_adapter"
                    if mapping.adapter
                    else "verified_mapping_has_no_symbol_or_adapter",
                }
            )
            continue
        consumed = registry.parameter_ids(mapping.adapter)
        ignored_parameters = sorted(
            parameter.id
            for parameter in manifest.parameters
            if parameter.scope == "authoring"
            and parameter.id not in consumed
            and parameter.id not in mapping.constructor_omissions
        )
        if ignored_parameters:
            ignored.append(
                {
                    "manifest": manifest.id,
                    "adapter": mapping.adapter,
                    "parameters": ignored_parameters,
                }
            )
    return unresolved, ignored


def _catalog_fallbacks(catalog: NodeCatalog) -> list[dict[str, str]]:
    fallbacks: list[dict[str, str]] = []
    for backend in NATIVE_BACKENDS:
        for manifest in catalog.manifests:
            if manifest.deprecated:
                continue
            mapping = manifest.implementations.get(backend)
            if mapping is not None and mapping.adapter and mapping.adapter.startswith("keras_"):
                fallbacks.append(
                    {
                        "backend": backend,
                        "manifest": manifest.id,
                        "adapter": mapping.adapter,
                        "reason": "native_mapping_names_keras_adapter",
                    }
                )
    return fallbacks


def _source_fallbacks() -> list[dict[str, str]]:
    exporter_root = Path(__file__).resolve().parents[1]
    fallbacks: list[dict[str, str]] = []
    for filename in (
        "native_tensorflow_exporter.py",
        "pytorch_exporter.py",
        "native_jax_exporter.py",
    ):
        path = exporter_root / filename
        if "KerasSourceExporter" in path.read_text(encoding="utf-8"):
            fallbacks.append(
                {
                    "backend": filename.removesuffix("_exporter.py"),
                    "source": filename,
                    "reason": "native_exporter_imports_keras_exporter",
                }
            )
    return fallbacks


def build_catalog_adapter_audit(
    catalog: NodeCatalog,
    registries: dict[str, NodeEmitterRegistry],
) -> dict[str, Any]:
    """Build an evidence report without promoting catalog mappings to execution proof."""

    backends: dict[str, Any] = {}
    for backend in BACKENDS:
        registry = registries[backend]
        unresolved, ignored = _adapter_records(catalog, backend, registry)
        backends[backend] = {
            "active_manifests": sum(not manifest.deprecated for manifest in catalog.manifests),
            "catalog_mapping_counts": _mapping_counts(catalog, backend),
            "registered_adapters": list(registry.adapter_ids),
            "registered_runtime_adapters": [
                adapter_id
                for adapter_id in registry.adapter_ids
                if registry.has_runtime(adapter_id)
            ],
            "unresolved_adapters": unresolved,
            "ignored_editable_parameters": ignored,
        }
    return {
        "evidence_basis": "declarative_catalog_and_registered_emitters_only",
        "execution_proof": False,
        "active_manifest_count": sum(
            not manifest.deprecated for manifest in catalog.manifests
        ),
        "backends": backends,
        "native_to_keras_fallbacks": [
            *_catalog_fallbacks(catalog),
            *_source_fallbacks(),
        ],
    }
