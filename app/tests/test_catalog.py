from common.catalogs.loader import get_node_catalog


def test_catalog_contains_large_manifest_set() -> None:
    catalog = get_node_catalog()
    assert len(catalog.manifests) >= 80
    dense = catalog.get("layer.dense", 1)
    assert dense.display_name == "Dense / Linear"
    assert {item.id for item in dense.parameters} >= {"units", "activation", "use_bias"}
