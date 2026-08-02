from __future__ import annotations

from fastapi import APIRouter

from configurations.settings import get_settings

router = APIRouter(prefix="/api/metadata", tags=["metadata"])


@router.get("")
def metadata() -> dict[str, object]:
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "schema_version": 1,
        "frameworks": ["keras", "tensorflow_keras", "pytorch"],
        "configuration": {
            "optimizers": ["adam", "adamw", "sgd", "rmsprop"],
            "losses": [
                "categorical_crossentropy",
                "sparse_categorical_crossentropy",
                "binary_crossentropy",
                "mean_squared_error",
                "mean_absolute_error",
            ],
            "metrics": ["accuracy", "precision", "recall", "auc", "mae", "mse"],
            "initializers": ["glorot_uniform", "he_normal", "zeros", "ones"],
            "regularizers": ["none", "l1", "l2", "l1_l2"],
        },
    }
