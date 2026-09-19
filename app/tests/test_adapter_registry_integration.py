from __future__ import annotations

import pytest
from common.catalogs.loader import get_node_catalog
from exporters.node_emitters import registry_for_backend
from exporters.node_emitters.contracts import EmissionResult, NodeEmitterRegistry

MANAGED_ADAPTERS = {"constant", "parameter", "average", "split", "detection"}


@pytest.mark.parametrize("backend", ["keras", "tensorflow", "pytorch", "jax"])
def test_registered_universal_adapters_cover_all_editable_parameters(backend: str) -> None:
    registry_for_backend(backend).validate_catalog(
        get_node_catalog(), adapter_ids=MANAGED_ADAPTERS
    )


def test_registry_rejects_duplicate_adapter_ids() -> None:
    registry = NodeEmitterRegistry("test")

    registry.register("known", lambda _context: EmissionResult())
    with pytest.raises(ValueError, match="Duplicate node emitter adapter"):
        registry.register("known", lambda _context: EmissionResult())


def test_registry_reports_unresolved_verified_mapping() -> None:
    registry = NodeEmitterRegistry("keras")

    with pytest.raises(ValueError, match="requires adapter 'constant'"):
        registry.validate_catalog(get_node_catalog(), adapter_ids={"constant"})


def test_registry_reports_ignored_editable_parameters() -> None:
    registry = NodeEmitterRegistry("keras")
    registry.register("constant", lambda _context: EmissionResult())

    with pytest.raises(ValueError, match="ignores editable parameters"):
        registry.validate_catalog(get_node_catalog(), adapter_ids={"constant"})
