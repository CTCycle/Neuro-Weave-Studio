# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import pytest

from common.catalogs.loader import get_node_catalog
from exporters.node_emitters import build_catalog_adapter_audit, registry_for_backend
from exporters.node_emitters.contracts import EmissionResult, NodeEmitterRegistry

MANAGED_ADAPTERS = {"constant", "parameter", "average", "split", "detection"}


###############################################################################
@pytest.mark.parametrize("backend", ["keras", "tensorflow", "pytorch", "jax"])
def test_registered_universal_adapters_cover_all_editable_parameters(backend: str) -> None:
    registry_for_backend(backend).validate_catalog(
        get_node_catalog(), adapter_ids=MANAGED_ADAPTERS
    )


###############################################################################
def test_keras_runtime_registry_covers_universal_runtime_adapters() -> None:
    registry = registry_for_backend("keras")

    assert all(
        registry.has_runtime(adapter)
        for adapter in (
            "constant",
            "parameter",
            "index",
            "average",
            "split",
            "detection",
        )
    )


###############################################################################
@pytest.mark.parametrize("backend", ["keras", "tensorflow", "pytorch", "jax"])
def test_source_exporters_use_authoritative_manifest_adapters(backend: str) -> None:
    registry = registry_for_backend(backend)

    assert all(
        registry.has_authoritative(adapter)
        for adapter in ("constant", "parameter", "index", "average", "split", "detection")
    )


###############################################################################
def test_source_exporters_no_longer_use_universal_node_id_allow_lists() -> None:
    exporter_root = __import__("pathlib").Path(__file__).parents[1] / "server" / "exporters"
    for filename in (
        "keras_exporter.py",
        "native_tensorflow_exporter.py",
        "pytorch_exporter.py",
        "native_jax_exporter.py",
    ):
        assert "registry_managed" not in (exporter_root / filename).read_text(encoding="utf-8")


###############################################################################
def test_registry_rejects_duplicate_adapter_ids() -> None:
    registry = NodeEmitterRegistry("test")

    registry.register("known", lambda _context: EmissionResult())
    with pytest.raises(ValueError, match="Duplicate node emitter adapter"):
        registry.register("known", lambda _context: EmissionResult())


###############################################################################
def test_registry_reports_unresolved_verified_mapping() -> None:
    registry = NodeEmitterRegistry("keras")

    with pytest.raises(ValueError, match="requires authoritative adapter 'constant'"):
        registry.validate_catalog(get_node_catalog(), adapter_ids={"constant"})


###############################################################################
def test_registry_reports_ignored_editable_parameters() -> None:
    registry = NodeEmitterRegistry("keras")
    registry.register("constant", lambda _context: EmissionResult(), authoritative=True)

    with pytest.raises(ValueError, match="ignores editable parameters"):
        registry.validate_catalog(get_node_catalog(), adapter_ids={"constant"})


###############################################################################
def test_catalog_adapter_audit_keeps_mapping_counts_separate_from_execution_proof() -> None:
    report = build_catalog_adapter_audit(
        get_node_catalog(),
        {
            backend: registry_for_backend(backend)
            for backend in ("keras", "tensorflow", "pytorch", "jax")
        },
    )

    assert report["evidence_basis"] == "declarative_catalog_and_registered_emitters_only"
    assert report["execution_proof"] is False
    assert report["active_manifest_count"] == 149
    for backend in ("keras", "tensorflow", "pytorch", "jax"):
        backend_report = report["backends"][backend]
        assert backend_report["active_manifests"] == 149
        assert backend_report["catalog_mapping_counts"] == {
            "verified": 149,
            "unsupported": 0,
            "unverified": 0,
            "missing": 0,
        }

    native_unresolved = report["backends"]["tensorflow"]["unresolved_adapters"]
    unresolved_ids = {item["adapter"] for item in native_unresolved}
    assert {"conv1d", "conv2d", "conv3d"}.isdisjoint(unresolved_ids)
    assert "depthwise" in unresolved_ids
    assert all(
        not backend_report["ignored_editable_parameters"]
        for backend_report in report["backends"].values()
    )
    assert report["native_to_keras_fallbacks"] == []
