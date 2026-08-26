from domain.compilation_catalog import get_compilation_catalog
from domain.graph import ModelConfiguration, ModelGraph
from validation.compile_configuration import validate_model_configuration


###############################################################################
def test_compilation_catalog_describes_runtime_supported_options() -> None:
    catalog = get_compilation_catalog()

    assert {item.id for item in catalog.optimizers} == {"adam", "adamw", "sgd", "rmsprop"}
    assert {item.id for item in catalog.losses} >= {
        "huber",
        "kullback_leibler_divergence",
    }
    assert any(parameter.id == "weight_decay" for item in catalog.optimizers if item.id == "adamw" for parameter in item.parameters)
    assert all(item.description for item in [*catalog.optimizers, *catalog.losses, *catalog.metrics])


###############################################################################
def test_extended_optimizer_parameters_share_validation_boundary() -> None:
    graph = ModelGraph(
        model_configuration=ModelConfiguration(
            optimizer="adamw",
            optimizer_parameters={
                "learning_rate": 0.001,
                "weight_decay": 0.004,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-7,
            },
        )
    )

    assert validate_model_configuration(graph, keras_enabled=True) == []
