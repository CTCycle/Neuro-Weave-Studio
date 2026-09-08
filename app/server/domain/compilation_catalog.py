from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

CompilationParameterType = Literal["number", "integer", "boolean", "enum"]
CompilationTask = Literal["regression", "binary_classification", "multiclass_classification"]
TargetContract = Literal[
    "continuous_same_shape",
    "binary_same_shape",
    "one_hot_same_shape",
    "class_index",
    "probability_distribution_same_shape",
]


###############################################################################
class CompilationParameter(BaseModel):
    id: str
    label: str
    type: CompilationParameterType
    default: float | int | bool | str
    description: str
    options: list[str] = Field(default_factory=list)
    constraints: dict[str, float | int | bool | str] = Field(default_factory=dict)


###############################################################################
class CompilationImplementation(BaseModel):
    supported: bool = False
    symbol: str | None = None
    adapter: str | None = None
    parameter_mapping: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None


###############################################################################
class CompilationOption(BaseModel):
    id: str
    label: str
    description: str
    parameters: list[CompilationParameter] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    formula: str | None = None
    implementations: dict[str, CompilationImplementation] = Field(default_factory=dict)
    supported_tasks: list[CompilationTask] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    target_contract: TargetContract | None = None
    activation_contract: list[str] = Field(default_factory=list)
    origin: Literal["builtin", "custom"] = "builtin"
    locked: bool = True
    can_edit: bool = False
    can_delete: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


###############################################################################
class CompilationCatalog(BaseModel):
    optimizers: list[CompilationOption] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    losses: list[CompilationOption] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    metrics: list[CompilationOption] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    initializers: list[str] = Field(default_factory=list)
    regularizers: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


###############################################################################
class CompilationCatalogProvider(Protocol):
    """Boundary for the future user-editable compilation catalog."""

    # -------------------------------------------------------------------------
    def get(self) -> CompilationCatalog: ...


###############################################################################
class StaticCompilationCatalogProvider:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._catalog: CompilationCatalog | None = None

    # -------------------------------------------------------------------------
    def get(self) -> CompilationCatalog:
        if self._catalog is None:
            self._catalog = _build_compilation_catalog()
        return self._catalog


###############################################################################
def _number(
    identifier: str,
    label: str,
    default: float,
    description: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> CompilationParameter:
    constraints: dict[str, float | int | bool | str] = {}
    if minimum is not None:
        constraints["min"] = minimum
    if maximum is not None:
        constraints["max"] = maximum
    return CompilationParameter(
        id=identifier,
        label=label,
        type="number",
        default=default,
        description=description,
        constraints=constraints,
    )


###############################################################################
def _optimizer(
    identifier: str,
    label: str,
    description: str,
    parameters: list[CompilationParameter],
) -> CompilationOption:
    symbols = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD", "rmsprop": "RMSprop"}
    return CompilationOption(
        id=identifier,
        label=label,
        description=description,
        parameters=parameters,
        supported_tasks=["regression", "binary_classification", "multiclass_classification"],
        implementations={
            "keras": CompilationImplementation(supported=True, symbol=symbols[identifier]),
            "pytorch": CompilationImplementation(
                supported=True, adapter=f"torch.optim.{symbols[identifier]}"
            ),
        },
    )


###############################################################################
def _mapped_option(
    identifier: str,
    label: str,
    description: str,
    *,
    keras_symbol: str,
    pytorch_adapter: str,
    supported_tasks: list[CompilationTask],
    target_contract: TargetContract,
    activation_contract: list[str] | None = None,
    formula: str | None = None,
) -> CompilationOption:
    return CompilationOption(
        id=identifier,
        label=label,
        description=description,
        formula=formula,
        implementations={
            "keras": CompilationImplementation(supported=True, symbol=keras_symbol),
            "pytorch": CompilationImplementation(supported=True, adapter=pytorch_adapter),
        },
        supported_tasks=supported_tasks,
        target_contract=target_contract,
        activation_contract=activation_contract or [],
        origin="builtin",
        locked=True,
        can_edit=False,
        can_delete=False,
    )


###############################################################################
def _build_compilation_catalog() -> CompilationCatalog:
    return CompilationCatalog(
        optimizers=[
            _optimizer(
                "adam",
                "Adam",
                "Adaptive first-order optimizer with momentum estimates.",
                [
                    _number(
                        "learning_rate",
                        "Learning rate",
                        0.001,
                        "Step size for each update.",
                        minimum=0.0,
                    ),
                    _number(
                        "beta_1",
                        "Beta 1",
                        0.9,
                        "Exponential decay for the first moment.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _number(
                        "beta_2",
                        "Beta 2",
                        0.999,
                        "Exponential decay for the second moment.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                ],
            ),
            _optimizer(
                "adamw",
                "AdamW",
                "Adam with decoupled weight decay.",
                [
                    _number(
                        "learning_rate",
                        "Learning rate",
                        0.001,
                        "Step size for each update.",
                        minimum=0.0,
                    ),
                    _number(
                        "weight_decay",
                        "Weight decay",
                        0.004,
                        "Decoupled regularization coefficient.",
                        minimum=0.0,
                    ),
                    _number(
                        "beta_1",
                        "Beta 1",
                        0.9,
                        "Exponential decay for the first moment.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _number(
                        "beta_2",
                        "Beta 2",
                        0.999,
                        "Exponential decay for the second moment.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                ],
            ),
            _optimizer(
                "sgd",
                "SGD",
                "Stochastic gradient descent with optional momentum.",
                [
                    _number(
                        "learning_rate",
                        "Learning rate",
                        0.01,
                        "Step size for each update.",
                        minimum=0.0,
                    ),
                    _number(
                        "momentum",
                        "Momentum",
                        0.0,
                        "Momentum contribution from the previous update.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    CompilationParameter(
                        id="nesterov",
                        label="Nesterov",
                        type="boolean",
                        default=False,
                        description="Use Nesterov momentum.",
                    ),
                ],
            ),
            _optimizer(
                "rmsprop",
                "RMSprop",
                "Adaptive optimizer using a moving average of squared gradients.",
                [
                    _number(
                        "learning_rate",
                        "Learning rate",
                        0.001,
                        "Step size for each update.",
                        minimum=0.0,
                    ),
                    _number(
                        "rho",
                        "Rho",
                        0.9,
                        "Discount factor for the moving average.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _number(
                        "momentum",
                        "Momentum",
                        0.0,
                        "Momentum contribution from the previous update.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                    CompilationParameter(
                        id="centered",
                        label="Centered",
                        type="boolean",
                        default=False,
                        description="Normalize gradients using their centered variance.",
                    ),
                ],
            ),
        ],
        losses=[
            _mapped_option(
                "mean_squared_error",
                "Mean squared error",
                "Squared difference between targets and predictions.",
                keras_symbol="MeanSquaredError",
                pytorch_adapter="mean_squared_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2",
            ),
            _mapped_option(
                "mean_absolute_error",
                "Mean absolute error",
                "Absolute difference between targets and predictions.",
                keras_symbol="MeanAbsoluteError",
                pytorch_adapter="mean_absolute_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}\lvert y_i-\hat y_i\rvert",
            ),
            _mapped_option(
                "binary_crossentropy",
                "Binary crossentropy",
                "Crossentropy for binary targets.",
                keras_symbol="BinaryCrossentropy",
                pytorch_adapter="binary_crossentropy",
                supported_tasks=["binary_classification"],
                target_contract="binary_same_shape",
                activation_contract=["sigmoid"],
                formula=r"\operatorname{BCE}=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(\hat p_i)+(1-y_i)\log(1-\hat p_i)\right]",
            ),
            _mapped_option(
                "categorical_crossentropy",
                "Categorical crossentropy",
                "Crossentropy for one-hot categorical targets.",
                keras_symbol="CategoricalCrossentropy",
                pytorch_adapter="categorical_crossentropy",
                supported_tasks=["multiclass_classification"],
                target_contract="one_hot_same_shape",
                activation_contract=["softmax"],
                formula=r"\operatorname{CCE}=-\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}y_{ic}\log(\hat p_{ic})",
            ),
            _mapped_option(
                "sparse_categorical_crossentropy",
                "Sparse categorical crossentropy",
                "Crossentropy for integer categorical targets.",
                keras_symbol="SparseCategoricalCrossentropy",
                pytorch_adapter="sparse_categorical_crossentropy",
                supported_tasks=["multiclass_classification"],
                target_contract="class_index",
                activation_contract=["softmax"],
                formula=r"\operatorname{SCCE}=-\frac{1}{n}\sum_{i=1}^{n}\log(\hat p_{i,y_i})",
            ),
            _mapped_option(
                "huber",
                "Huber",
                "Quadratic near zero and linear for large residuals.",
                keras_symbol="Huber",
                pytorch_adapter="huber",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"L_\delta(a)=\begin{cases}\frac{1}{2}a^2,&\lvert a\rvert\leq\delta\\\delta\left(\lvert a\rvert-\frac{1}{2}\delta\right),&\text{otherwise}\end{cases}",
            ),
            _mapped_option(
                "kullback_leibler_divergence",
                "Kullback-Leibler divergence",
                "Divergence between target and predicted distributions.",
                keras_symbol="KLDivergence",
                pytorch_adapter="kullback_leibler_divergence",
                supported_tasks=["multiclass_classification"],
                target_contract="probability_distribution_same_shape",
                activation_contract=["softmax"],
                formula=r"D_{KL}(P\parallel Q)=\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}P_{ic}\log\frac{P_{ic}}{Q_{ic}}",
            ),
        ],
        metrics=[
            _mapped_option(
                "accuracy",
                "Accuracy",
                "Correct predictions divided by total predictions.",
                keras_symbol="Accuracy",
                pytorch_adapter="accuracy",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="one_hot_same_shape",
                formula=r"\operatorname{Accuracy}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[\hat y_i=y_i]",
            ),
            _mapped_option(
                "binary_accuracy",
                "Binary accuracy",
                "Accuracy for binary predictions.",
                keras_symbol="BinaryAccuracy",
                pytorch_adapter="binary_accuracy",
                supported_tasks=["binary_classification"],
                target_contract="binary_same_shape",
                formula=r"\operatorname{BinaryAccuracy}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[\mathbf{1}[\hat p_i\geq\tau]=y_i]",
            ),
            _mapped_option(
                "categorical_accuracy",
                "Categorical accuracy",
                "Accuracy for one-hot categorical predictions.",
                keras_symbol="CategoricalAccuracy",
                pytorch_adapter="categorical_accuracy",
                supported_tasks=["multiclass_classification"],
                target_contract="one_hot_same_shape",
                formula=r"\operatorname{CategoricalAccuracy}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[\arg\max_c\hat p_{ic}=\arg\max_c y_{ic}]",
            ),
            _mapped_option(
                "sparse_categorical_accuracy",
                "Sparse categorical accuracy",
                "Accuracy for integer categorical predictions.",
                keras_symbol="SparseCategoricalAccuracy",
                pytorch_adapter="sparse_categorical_accuracy",
                supported_tasks=["multiclass_classification"],
                target_contract="class_index",
                formula=r"\operatorname{SparseAccuracy}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[\arg\max_c\hat p_{ic}=y_i]",
            ),
            _mapped_option(
                "precision",
                "Precision",
                "Positive predictions that are correct.",
                keras_symbol="Precision",
                pytorch_adapter="precision",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="binary_same_shape",
                formula=r"\operatorname{Precision}=\frac{TP}{TP+FP}",
            ),
            _mapped_option(
                "recall",
                "Recall",
                "Actual positives that are identified.",
                keras_symbol="Recall",
                pytorch_adapter="recall",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="binary_same_shape",
                formula=r"\operatorname{Recall}=\frac{TP}{TP+FN}",
            ),
            _mapped_option(
                "auc",
                "AUC",
                "Area under the receiver operating characteristic curve.",
                keras_symbol="AUC",
                pytorch_adapter="auc",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="binary_same_shape",
                formula=r"\operatorname{AUC}=\Pr(\hat p^+>\hat p^-)+\frac{1}{2}\Pr(\hat p^+=\hat p^-)",
            ),
            _mapped_option(
                "mean_absolute_error",
                "Mean absolute error",
                "Mean absolute prediction error.",
                keras_symbol="MeanAbsoluteError",
                pytorch_adapter="mean_absolute_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}\lvert y_i-\hat y_i\rvert",
            ),
            _mapped_option(
                "mean_squared_error",
                "Mean squared error",
                "Mean squared prediction error.",
                keras_symbol="MeanSquaredError",
                pytorch_adapter="mean_squared_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2",
            ),
            _mapped_option(
                "root_mean_squared_error",
                "Root mean squared error",
                "Square root of mean squared prediction error.",
                keras_symbol="RootMeanSquaredError",
                pytorch_adapter="root_mean_squared_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}",
            ),
        ],
        initializers=[
            "uniform",
            "glorot_uniform",
            "glorot_normal",
            "he_uniform",
            "he_normal",
            "zeros",
            "ones",
            "random_normal",
            "random_uniform",
            "orthogonal",
        ],
        regularizers=["none", "l1", "l2", "l1_l2"],
        constraints=["none", "max_norm", "min_max_norm", "non_neg", "unit_norm"],
    )


_COMPILATION_CATALOG_PROVIDER = StaticCompilationCatalogProvider()


###############################################################################
def get_compilation_catalog_provider() -> CompilationCatalogProvider:
    return _COMPILATION_CATALOG_PROVIDER


class PersistentCompilationCatalogProvider:
    """Merge server-owned built-ins with installation-wide user definitions."""

    def __init__(self, uow_factory: Callable[[], Any]) -> None:
        self._uow_factory = uow_factory

    def get(self) -> CompilationCatalog:
        # Keep the import local so the static catalog remains usable by pure
        # compiler tests without importing the persistence layer.
        with self._uow_factory() as uow:
            records = [
                (
                    record.id,
                    record.kind,
                    record.label,
                    record.description,
                    record.formula_latex,
                    record.created_at,
                    record.updated_at,
                )
                for record in uow.metrics.list_all()
            ]
        catalog = get_compilation_catalog().model_copy(deep=True)
        custom_by_kind: dict[str, list[CompilationOption]] = {"metric": [], "loss": []}
        for metric_id, kind, label, description, formula, created_at, updated_at in records:
            option = CompilationOption(
                id=metric_id,
                label=label,
                description=description,
                formula=formula,
                implementations={
                    "keras": CompilationImplementation(supported=True, adapter="custom"),
                    "pytorch": CompilationImplementation(supported=True, adapter="custom"),
                },
                supported_tasks=[
                    "regression",
                    "binary_classification",
                    "multiclass_classification",
                ],
                target_contract=None,
                origin="custom",
                locked=False,
                can_edit=True,
                can_delete=True,
                created_at=created_at,
                updated_at=updated_at,
            )
            custom_by_kind.setdefault(kind, []).append(option)
        return catalog.model_copy(
            update={
                "metrics": [*catalog.metrics, *custom_by_kind["metric"]],
                "losses": [*catalog.losses, *custom_by_kind["loss"]],
            }
        )


###############################################################################
def get_compilation_catalog() -> CompilationCatalog:
    return get_compilation_catalog_provider().get()


###############################################################################
def get_compilation_catalog_for_implementation(
    catalog: CompilationCatalog,
    implementation: str,
) -> CompilationCatalog:
    """Filter the trusted catalog to options with an explicit implementation."""

    def supported(options: list[CompilationOption]) -> list[CompilationOption]:
        return [
            option
            for option in options
            if option.implementations.get(implementation, CompilationImplementation()).supported
        ]

    return catalog.model_copy(
        deep=True,
        update={
            "optimizers": supported(catalog.optimizers),
            "losses": supported(catalog.losses),
            "metrics": supported(catalog.metrics),
        },
    )
