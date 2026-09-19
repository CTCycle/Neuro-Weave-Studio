from __future__ import annotations

import json
from pathlib import Path

import pytest
from common.catalogs import loader
from common.catalogs.loader import (
    RESOURCE_ROOT,
    _catalog_extensions,
    _expand_parameter_groups,
    _parameter_group_catalog,
    _parameter_surface_catalog,
    get_node_catalog,
)


def _resource_node_ids() -> set[str]:
    identifiers: set[str] = set()
    for path in RESOURCE_ROOT.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else raw["nodes"]
        identifiers.update(
            str(item["id"])
            for item in items
            if str(item.get("id", "")).startswith(
                ("layer.", "conv.", "recurrent.", "attention.", "transformer.")
            )
        )
    return identifiers


def test_node_parameter_surfaces_are_declarative_and_match_catalog() -> None:
    catalog = get_node_catalog()
    groups, assignments = _parameter_group_catalog(_catalog_extensions())
    surfaces = _parameter_surface_catalog()
    expected_nodes = _resource_node_ids()

    assert expected_nodes == set(surfaces)
    source = Path(loader.__file__).read_text(encoding="utf-8")
    assert "_safe_layer_parameters" not in source
    assert "_ACTIVATION_OPTIONS" not in source
    assert "_DTYPE_OPTIONS" not in source

    for node_id in sorted(expected_nodes):
        manifest = catalog.get(node_id)
        group_ids = {
            str(parameter["id"])
            for group_id in assignments.get(node_id, [])
            for parameter in groups[group_id]
        }
        surface_ids = [str(parameter["id"]) for parameter in surfaces[node_id]]
        manifest_ids = [parameter.id for parameter in manifest.parameters]

        assert len(manifest_ids) == len(set(manifest_ids))
        assert set(manifest_ids) == group_ids | set(surface_ids)
        assert manifest.parameter_groups == assignments.get(node_id, [])


def test_parameter_group_expansion_preserves_declarative_metadata() -> None:
    parameter = {
        "id": "scale",
        "display_name": "Scale",
        "type": "number",
        "scope": "runtime",
        "aliases": ["gain"],
        "enabled_when": {"enabled": True},
        "value_schema": {"type": "number", "nullable": True},
    }
    expanded = _expand_parameter_groups(
        {"id": "example.node", "parameter_groups": ["shared"]},
        {"shared": [parameter]},
    )

    assert expanded["parameters"] == [parameter]


def test_parameter_group_expansion_rejects_group_and_surface_collisions() -> None:
    with pytest.raises(RuntimeError, match="Parameter surface collision"):
        _expand_parameter_groups(
            {
                "id": "example.node",
                "parameter_groups": ["shared"],
                "parameters": [{"id": "scale", "type": "number"}],
            },
            {
                "shared": [
                    {"id": "scale", "type": "number"},
                ]
            },
        )


def test_parameter_group_expansion_rejects_alias_collisions() -> None:
    with pytest.raises(RuntimeError, match="alias collision"):
        _expand_parameter_groups(
            {"id": "example.node", "parameter_groups": ["shared", "other"]},
            {
                "shared": [{"id": "scale", "aliases": ["gain"]}],
                "other": [{"id": "gain"}],
            },
        )
