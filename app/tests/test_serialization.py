from domain.graph import ModelGraph, ProjectCreate, ProjectExportBundle

###############################################################################
def test_project_bundle_round_trip() -> None:
    bundle = ProjectExportBundle(project=ProjectCreate(name="Example", graph=ModelGraph()))
    restored = ProjectExportBundle.model_validate_json(bundle.model_dump_json())
    assert restored.format_version == 3
    assert restored.project.graph.schema_version == 2
    assert restored.project.name == "Example"
