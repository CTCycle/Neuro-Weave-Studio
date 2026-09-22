from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from domain.manifests import ParameterDefinition, ParameterValueSchema

CompilationTask = Literal["regression", "binary_classification", "multiclass_classification"]
TargetContract = Literal[
    "continuous_same_shape",
    "binary_same_shape",
    "one_hot_same_shape",
    "class_index",
    "probability_distribution_same_shape",
]

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
    parameters: list[ParameterDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
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
) -> ParameterDefinition:
    constraints: dict[str, float | int | bool | str] = {}
    if minimum is not None:
        constraints["min"] = minimum
    if maximum is not None:
        constraints["max"] = maximum
    return ParameterDefinition(
        id=identifier,
        display_name=label,
        type="number",
        default=default,
        description=description,
        constraints=constraints,
    )


###############################################################################
def _boolean(
    identifier: str,
    label: str,
    default: bool,
    description: str,
) -> ParameterDefinition:
    return ParameterDefinition(
        id=identifier,
        display_name=label,
        type="boolean",
        default=default,
        description=description,
    )


###############################################################################
def _integer(
    identifier: str,
    label: str,
    default: int | None,
    description: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    nullable: bool = False,
) -> ParameterDefinition:
    constraints: dict[str, int | bool | str] = {}
    if minimum is not None:
        constraints["min"] = minimum
    if maximum is not None:
        constraints["max"] = maximum
    return ParameterDefinition(
        id=identifier,
        display_name=label,
        type="integer",
        default=default,
        nullable=nullable,
        description=description,
        constraints=constraints,
    )


###############################################################################
def _enum(
    identifier: str,
    label: str,
    default: str,
    options: list[str],
    description: str,
) -> ParameterDefinition:
    return ParameterDefinition(
        id=identifier,
        display_name=label,
        type="enum",
        default=default,
        options=options,
        description=description,
    )


###############################################################################
def _number_list(
    identifier: str,
    label: str,
    description: str,
) -> ParameterDefinition:
    return ParameterDefinition(
        id=identifier,
        display_name=label,
        type="list",
        default=None,
        nullable=True,
        description=description,
        value_schema=ParameterValueSchema(type="number"),
    )

###############################################################################
def _optimizer(
    identifier: str,
    label: str,
    description: str,
    parameters: list[ParameterDefinition],
    *,
    keras_symbol: str | None = None,
    pytorch_symbol: str | None = None,
    pytorch_parameter_mapping: dict[str, str] | None = None,
    tensorflow_symbol: str | None = None,
    tensorflow_parameter_mapping: dict[str, str] | None = None,
    jax_adapter: str | None = None,
    jax_parameter_mapping: dict[str, str] | None = None,
) -> CompilationOption:
    symbols = {
        "adam": "Adam",
        "adamw": "AdamW",
        "sgd": "SGD",
        "rmsprop": "RMSprop",
        "adagrad": "Adagrad",
        "adadelta": "Adadelta",
        "adamax": "Adamax",
        "nadam": "Nadam",
    }
    keras_symbol = keras_symbol or symbols[identifier]
    pytorch_symbol = pytorch_symbol or {
        "adam": "Adam",
        "adamw": "AdamW",
        "sgd": "SGD",
        "rmsprop": "RMSprop",
        "adagrad": "Adagrad",
        "adadelta": "Adadelta",
        "adamax": "Adamax",
        "nadam": "NAdam",
    }[identifier]
    tensorflow_symbol = tensorflow_symbol or keras_symbol
    implementations = {
        "keras": CompilationImplementation(supported=True, symbol=keras_symbol),
        "pytorch": CompilationImplementation(
            supported=True,
            adapter=f"torch.optim.{pytorch_symbol}",
            parameter_mapping=pytorch_parameter_mapping or {},
        ),
        "tensorflow": CompilationImplementation(
            supported=True,
            symbol=tensorflow_symbol,
            parameter_mapping=tensorflow_parameter_mapping or {},
        ),
    }
    if jax_adapter is not None:
        implementations["jax"] = CompilationImplementation(
            supported=True,
            adapter=jax_adapter,
            parameter_mapping=jax_parameter_mapping or {},
        )
    return CompilationOption(
        id=identifier,
        label=label,
        description=description,
        parameters=parameters,
        supported_tasks=["regression", "binary_classification", "multiclass_classification"],
        implementations=implementations,
    )

###############################################################################
def _mapped_option(
    identifier: str,
    label: str,
    description: str,
    *,
    keras_symbol: str | None,
    pytorch_adapter: str,
    supported_tasks: list[CompilationTask],
    target_contract: TargetContract,
    activation_contract: list[str] | None = None,
    formula: str | None = None,
    parameters: list[ParameterDefinition] | None = None,
    keras_parameter_mapping: dict[str, str] | None = None,
    keras_adapter: str | None = None,
    tensorflow_adapter: str | None = None,
    jax_adapter: str | None = None,
) -> CompilationOption:
    tensorflow_adapter = tensorflow_adapter or pytorch_adapter
    jax_adapter = jax_adapter or pytorch_adapter
    return CompilationOption(
        id=identifier,
        label=label,
        description=description,
        parameters=parameters or [],
        formula=formula,
        implementations={
            "keras": CompilationImplementation(
                supported=True,
                symbol=keras_symbol,
                adapter=keras_adapter,
                parameter_mapping=keras_parameter_mapping or {},
            ),
            "pytorch": CompilationImplementation(supported=True, adapter=pytorch_adapter),
            "tensorflow": CompilationImplementation(supported=True, adapter=tensorflow_adapter),
            "jax": CompilationImplementation(supported=True, adapter=jax_adapter),
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
                    _number("weight_decay", "Weight decay", 0.0, "L2 weight decay coefficient.", minimum=0.0),
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
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "weight_decay": "weight_decay",
                    "beta_1": "betas",
                    "beta_2": "betas",
                    "epsilon": "eps",
                },
                jax_adapter="adam",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "beta_1": "b1",
                    "beta_2": "b2",
                    "epsilon": "eps",
                },
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
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "weight_decay": "weight_decay",
                    "beta_1": "betas",
                    "beta_2": "betas",
                    "epsilon": "eps",
                },
                jax_adapter="adamw",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "weight_decay": "weight_decay",
                    "beta_1": "b1",
                    "beta_2": "b2",
                    "epsilon": "eps",
                },
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
                    _boolean("nesterov", "Nesterov", False, "Use Nesterov momentum."),
                    _number("weight_decay", "Weight decay", 0.0, "L2 weight decay coefficient.", minimum=0.0),
                ],
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "weight_decay": "weight_decay",
                },
                jax_adapter="sgd",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "momentum": "momentum",
                    "nesterov": "nesterov",
                },
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
                    _boolean(
                        "centered",
                        "Centered",
                        False,
                        "Normalize gradients using their centered variance.",
                    ),
                    _number("weight_decay", "Weight decay", 0.0, "L2 weight decay coefficient.", minimum=0.0),
                ],
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "rho": "alpha",
                    "epsilon": "eps",
                    "weight_decay": "weight_decay",
                },
                jax_adapter="rmsprop",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "rho": "decay",
                    "momentum": "momentum",
                    "epsilon": "eps",
                    "centered": "centered",
                },
            ),
            _optimizer(
                "adagrad",
                "Adagrad",
                "Adaptive optimizer with per-parameter learning rates.",
                [
                    _number("learning_rate", "Learning rate", 0.001, "Step size for each update.", minimum=0.0),
                    _number(
                        "initial_accumulator_value",
                        "Initial accumulator value",
                        0.1,
                        "Initial value for each squared-gradient accumulator.",
                        minimum=0.0,
                    ),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                    _number("weight_decay", "Weight decay", 0.0, "L2 weight decay coefficient.", minimum=0.0),
                ],
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "epsilon": "eps",
                },
                jax_adapter="adagrad",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "initial_accumulator_value": "initial_accumulator_value",
                    "epsilon": "eps",
                },
            ),
            _optimizer(
                "adadelta",
                "Adadelta",
                "Adaptive optimizer with a moving window of updates.",
                [
                    _number("learning_rate", "Learning rate", 0.001, "Step size for each update.", minimum=0.0),
                    _number("rho", "Rho", 0.95, "Discount factor for the moving average.", minimum=0.0, maximum=1.0),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                    _number("weight_decay", "Weight decay", 0.0, "L2 weight decay coefficient.", minimum=0.0),
                ],
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "epsilon": "eps",
                },
                jax_adapter="adadelta",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "rho": "rho",
                    "epsilon": "eps",
                    "weight_decay": "weight_decay",
                },
            ),
            _optimizer(
                "adamax",
                "Adamax",
                "Adam variant using the infinity norm.",
                [
                    _number("learning_rate", "Learning rate", 0.001, "Step size for each update.", minimum=0.0),
                    _number("beta_1", "Beta 1", 0.9, "Exponential decay for the first moment.", minimum=0.0, maximum=1.0),
                    _number("beta_2", "Beta 2", 0.999, "Exponential decay for the infinity norm.", minimum=0.0, maximum=1.0),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                    _number("weight_decay", "Weight decay", 0.0, "L2 weight decay coefficient.", minimum=0.0),
                ],
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "beta_1": "betas",
                    "beta_2": "betas",
                    "epsilon": "eps",
                },
                jax_adapter="adamax",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "beta_1": "b1",
                    "beta_2": "b2",
                    "epsilon": "eps",
                },
            ),
            _optimizer(
                "nadam",
                "Nadam",
                "Nesterov-accelerated adaptive moment estimation.",
                [
                    _number("learning_rate", "Learning rate", 0.001, "Step size for each update.", minimum=0.0),
                    _number("beta_1", "Beta 1", 0.9, "Exponential decay for the first moment.", minimum=0.0, maximum=1.0),
                    _number("beta_2", "Beta 2", 0.999, "Exponential decay for the second moment.", minimum=0.0, maximum=1.0),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                    _number("weight_decay", "Weight decay", 0.0, "L2 weight decay coefficient.", minimum=0.0),
                ],
                pytorch_parameter_mapping={
                    "learning_rate": "lr",
                    "beta_1": "betas",
                    "beta_2": "betas",
                    "epsilon": "eps",
                },
                jax_adapter="nadam",
                jax_parameter_mapping={
                    "learning_rate": "learning_rate",
                    "beta_1": "b1",
                    "beta_2": "b2",
                    "epsilon": "eps",
                },
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
                parameters=[
                    _boolean("from_logits", "From logits", False, "Interpret predictions as logits."),
                    _number(
                        "label_smoothing",
                        "Label smoothing",
                        0.0,
                        "Blend binary targets toward one half.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    ParameterDefinition(
                        id="axis",
                        display_name="Axis",
                        type="axis",
                        default=-1,
                        description="Axis used for the binary reduction.",
                    ),
                ],
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
                parameters=[
                    _boolean("from_logits", "From logits", False, "Interpret predictions as logits."),
                    _number(
                        "label_smoothing",
                        "Label smoothing",
                        0.0,
                        "Blend categorical targets toward a uniform distribution.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    ParameterDefinition(
                        id="axis",
                        display_name="Axis",
                        type="axis",
                        default=-1,
                        description="Class axis used for the categorical reduction.",
                    ),
                ],
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
                parameters=[
                    _boolean("from_logits", "From logits", False, "Interpret predictions as logits."),
                    _number(
                        "label_smoothing",
                        "Label smoothing",
                        0.0,
                        "Blend class targets toward a uniform distribution.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    ParameterDefinition(
                        id="axis",
                        display_name="Axis",
                        type="axis",
                        default=-1,
                        description="Class axis used for the sparse categorical reduction.",
                    ),
                    _integer(
                        "ignore_class",
                        "Ignore class",
                        None,
                        "Optional class index excluded from the reduction.",
                        nullable=True,
                    ),
                ],
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
                parameters=[
                    _number("delta", "Delta", 1.0, "Quadratic-to-linear transition point.", minimum=0.0)
                ],
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
            _mapped_option(
                "log_cosh",
                "LogCosh",
                "Smooth logarithmic hyperbolic-cosine regression loss.",
                keras_symbol="LogCosh",
                pytorch_adapter="log_cosh",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{LogCosh}(a)=\log(\cosh(a))",
            ),
            _mapped_option(
                "mean_absolute_percentage_error",
                "Mean absolute percentage error",
                "Absolute percentage error averaged over the target values.",
                keras_symbol="MeanAbsolutePercentageError",
                pytorch_adapter="mean_absolute_percentage_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MAPE}=\frac{100}{n}\sum_i\frac{|y_i-\hat y_i|}{|y_i|}",
            ),
            _mapped_option(
                "mean_squared_logarithmic_error",
                "Mean squared logarithmic error",
                "Squared error between logarithmic target and prediction values.",
                keras_symbol="MeanSquaredLogarithmicError",
                pytorch_adapter="mean_squared_logarithmic_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MSLE}=\frac{1}{n}\sum_i(\log(1+y_i)-\log(1+\hat y_i))^2",
            ),
            _mapped_option(
                "hinge",
                "Hinge",
                "Margin loss for binary targets encoded as -1 and 1.",
                keras_symbol="Hinge",
                pytorch_adapter="hinge",
                supported_tasks=["binary_classification"],
                target_contract="binary_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{Hinge}=\frac{1}{n}\sum_i\max(0,1-y_i\hat y_i)",
            ),
            _mapped_option(
                "squared_hinge",
                "Squared hinge",
                "Squared margin loss for binary targets encoded as -1 and 1.",
                keras_symbol="SquaredHinge",
                pytorch_adapter="squared_hinge",
                supported_tasks=["binary_classification"],
                target_contract="binary_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{SquaredHinge}=\frac{1}{n}\sum_i\max(0,1-y_i\hat y_i)^2",
            ),
            _mapped_option(
                "categorical_hinge",
                "Categorical hinge",
                "Multiclass margin loss comparing the positive class to the largest negative class.",
                keras_symbol="CategoricalHinge",
                pytorch_adapter="categorical_hinge",
                supported_tasks=["multiclass_classification"],
                target_contract="one_hot_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{CategoricalHinge}=\frac{1}{n}\sum_i\max(0,\max_{c:y_c=0}\hat y_c-y_{+}+1)",
            ),
            _mapped_option(
                "poisson",
                "Poisson",
                "Poisson negative log-likelihood for non-negative regression targets.",
                keras_symbol="Poisson",
                pytorch_adapter="poisson",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{Poisson}=\frac{1}{n}\sum_i(\hat y_i-y_i\log(\hat y_i))",
            ),
            _mapped_option(
                "dice",
                "Dice",
                "Overlap loss based on the Dice coefficient.",
                keras_symbol="Dice",
                pytorch_adapter="dice",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="one_hot_same_shape",
                activation_contract=["sigmoid", "softmax"],
                formula=r"\operatorname{Dice}=1-\frac{2\sum_i y_i\hat y_i}{\sum_i y_i+\sum_i\hat y_i}",
                parameters=[
                    ParameterDefinition(
                        id="axis",
                        display_name="Axis",
                        type="axis",
                        default=None,
                        nullable=True,
                        description="Optional reduction axis for the overlap calculation.",
                    )
                ],
            ),
            _mapped_option(
                "tversky",
                "Tversky",
                "Asymmetric overlap loss with false-positive and false-negative weights.",
                keras_symbol="Tversky",
                pytorch_adapter="tversky",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="one_hot_same_shape",
                activation_contract=["sigmoid", "softmax"],
                formula=r"\operatorname{Tversky}=1-\frac{TP}{TP+\alpha FP+\beta FN}",
                parameters=[
                    _number("alpha", "Alpha", 0.5, "False-positive weight.", minimum=0.0, maximum=1.0),
                    _number("beta", "Beta", 0.5, "False-negative weight.", minimum=0.0, maximum=1.0),
                    ParameterDefinition(
                        id="axis",
                        display_name="Axis",
                        type="axis",
                        default=None,
                        nullable=True,
                        description="Optional reduction axis for the overlap calculation.",
                    ),
                ],
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
                "top_k_categorical_accuracy",
                "Top-k categorical accuracy",
                "Fraction of one-hot categorical targets found among the top-k predictions.",
                keras_symbol="TopKCategoricalAccuracy",
                pytorch_adapter="top_k_categorical_accuracy",
                supported_tasks=["multiclass_classification"],
                target_contract="one_hot_same_shape",
                formula=r"\operatorname{TopKAccuracy}=\frac{1}{n}\sum_i\mathbf{1}[y_i\in\operatorname{TopK}(\hat y_i)]",
                parameters=[
                    ParameterDefinition(
                        id="k",
                        display_name="Top-k",
                        type="integer",
                        default=5,
                        constraints={"min": 1},
                        description="Number of highest-scoring classes to consider correct.",
                    )
                ],
            ),
            _mapped_option(
                "sparse_top_k_categorical_accuracy",
                "Sparse top-k categorical accuracy",
                "Fraction of integer class targets found among the top-k predictions.",
                keras_symbol="SparseTopKCategoricalAccuracy",
                pytorch_adapter="sparse_top_k_categorical_accuracy",
                supported_tasks=["multiclass_classification"],
                target_contract="class_index",
                formula=r"\operatorname{SparseTopKAccuracy}=\frac{1}{n}\sum_i\mathbf{1}[y_i\in\operatorname{TopK}(\hat y_i)]",
                parameters=[
                    ParameterDefinition(
                        id="k",
                        display_name="Top-k",
                        type="integer",
                        default=5,
                        constraints={"min": 1},
                        description="Number of highest-scoring classes to consider correct.",
                    )
                ],
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
                parameters=[
                    _number(
                        "threshold",
                        "Threshold",
                        0.5,
                        "Positive prediction threshold.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _integer(
                        "top_k",
                        "Top-k",
                        None,
                        "Optional number of highest-scoring predictions to treat as positive.",
                        minimum=1,
                        nullable=True,
                    ),
                    _integer(
                        "class_id",
                        "Class ID",
                        None,
                        "Optional class index for a class-specific metric.",
                        minimum=0,
                        nullable=True,
                    ),
                ],
                keras_parameter_mapping={"threshold": "thresholds"},
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
                parameters=[
                    _number(
                        "threshold",
                        "Threshold",
                        0.5,
                        "Positive prediction threshold.",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    _integer(
                        "top_k",
                        "Top-k",
                        None,
                        "Optional number of highest-scoring predictions to treat as positive.",
                        minimum=1,
                        nullable=True,
                    ),
                    _integer(
                        "class_id",
                        "Class ID",
                        None,
                        "Optional class index for a class-specific metric.",
                        minimum=0,
                        nullable=True,
                    ),
                ],
                keras_parameter_mapping={"threshold": "thresholds"},
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
                parameters=[
                    _integer(
                        "num_thresholds",
                        "Number of thresholds",
                        200,
                        "Number of thresholds used to approximate the curve.",
                        minimum=1,
                    ),
                    _enum(
                        "curve",
                        "Curve",
                        "ROC",
                        ["ROC", "PR"],
                        "Curve family used for the area calculation.",
                    ),
                    _enum(
                        "summation_method",
                        "Summation method",
                        "interpolation",
                        ["interpolation", "minoring", "majoring"],
                        "How adjacent curve points are integrated.",
                    ),
                    _number_list(
                        "thresholds",
                        "Thresholds",
                        "Optional explicit threshold values for the curve.",
                    ),
                    _boolean(
                        "multi_label",
                        "Multi-label",
                        False,
                        "Whether each final-axis label is evaluated independently.",
                    ),
                    _integer(
                        "num_labels",
                        "Number of labels",
                        None,
                        "Optional label count for multi-label predictions.",
                        minimum=1,
                        nullable=True,
                    ),
                    ParameterDefinition(
                        id="label_weights",
                        display_name="Label weights",
                        type="list",
                        default=None,
                        nullable=True,
                        description="Optional weights for each multi-label AUC.",
                        value_schema=ParameterValueSchema(type="number"),
                    ),
                    _boolean(
                        "from_logits",
                        "From logits",
                        False,
                        "Whether predictions should be passed through a sigmoid first.",
                    ),
                ],
            ),
            _mapped_option(
                "true_positives",
                "True positives",
                "Count of positive predictions that match positive targets.",
                keras_symbol="TruePositives",
                pytorch_adapter="true_positives",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="binary_same_shape",
                formula=r"TP=\sum_i\mathbf{1}[\hat y_i=1\land y_i=1]",
                parameters=[
                    _number("threshold", "Threshold", 0.5, "Positive prediction threshold.", minimum=0.0, maximum=1.0)
                ],
                keras_parameter_mapping={"threshold": "thresholds"},
            ),
            _mapped_option(
                "true_negatives",
                "True negatives",
                "Count of negative predictions that match negative targets.",
                keras_symbol="TrueNegatives",
                pytorch_adapter="true_negatives",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="binary_same_shape",
                formula=r"TN=\sum_i\mathbf{1}[\hat y_i=0\land y_i=0]",
                parameters=[
                    _number("threshold", "Threshold", 0.5, "Positive prediction threshold.", minimum=0.0, maximum=1.0)
                ],
                keras_parameter_mapping={"threshold": "thresholds"},
            ),
            _mapped_option(
                "false_positives",
                "False positives",
                "Count of positive predictions that disagree with negative targets.",
                keras_symbol="FalsePositives",
                pytorch_adapter="false_positives",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="binary_same_shape",
                formula=r"FP=\sum_i\mathbf{1}[\hat y_i=1\land y_i=0]",
                parameters=[
                    _number("threshold", "Threshold", 0.5, "Positive prediction threshold.", minimum=0.0, maximum=1.0)
                ],
                keras_parameter_mapping={"threshold": "thresholds"},
            ),
            _mapped_option(
                "false_negatives",
                "False negatives",
                "Count of negative predictions that disagree with positive targets.",
                keras_symbol="FalseNegatives",
                pytorch_adapter="false_negatives",
                supported_tasks=["binary_classification", "multiclass_classification"],
                target_contract="binary_same_shape",
                formula=r"FN=\sum_i\mathbf{1}[\hat y_i=0\land y_i=1]",
                parameters=[
                    _number("threshold", "Threshold", 0.5, "Positive prediction threshold.", minimum=0.0, maximum=1.0)
                ],
                keras_parameter_mapping={"threshold": "thresholds"},
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
            _mapped_option(
                "mean_absolute_percentage_error",
                "Mean absolute percentage error",
                "Mean absolute percentage prediction error.",
                keras_symbol="MeanAbsolutePercentageError",
                pytorch_adapter="mean_absolute_percentage_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MAPE}=\frac{100}{n}\sum_i\frac{|y_i-\hat y_i|}{|y_i|}",
            ),
            _mapped_option(
                "mean_squared_logarithmic_error",
                "Mean squared logarithmic error",
                "Mean squared logarithmic prediction error.",
                keras_symbol="MeanSquaredLogarithmicError",
                pytorch_adapter="mean_squared_logarithmic_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{MSLE}=\frac{1}{n}\sum_i(\log(1+y_i)-\log(1+\hat y_i))^2",
            ),
            _mapped_option(
                "cosine_similarity",
                "Cosine similarity",
                "Cosine similarity between target and prediction vectors.",
                keras_symbol="CosineSimilarity",
                pytorch_adapter="cosine_similarity",
                supported_tasks=["regression", "binary_classification", "multiclass_classification"],
                target_contract="continuous_same_shape",
                activation_contract=["linear", "sigmoid", "softmax"],
                formula=r"\operatorname{CosineSimilarity}=\frac{y\cdot\hat y}{\|y\|\|\hat y\|}",
                parameters=[
                    ParameterDefinition(
                        id="axis",
                        display_name="Axis",
                        type="axis",
                        default=-1,
                        description="Axis along which vectors are compared.",
                    )
                ],
            ),
            _mapped_option(
                "log_cosh_error",
                "Log-cosh error",
                "Smooth logarithmic hyperbolic-cosine prediction error.",
                keras_symbol="LogCoshError",
                pytorch_adapter="log_cosh_error",
                supported_tasks=["regression"],
                target_contract="continuous_same_shape",
                activation_contract=["linear"],
                formula=r"\operatorname{LogCoshError}=\frac{1}{n}\sum_i\log(\cosh(y_i-\hat y_i))",
            ),
            _mapped_option(
                "prediction_mean",
                "Prediction mean",
                "Mean of all values in the prediction tensor; target values are ignored.",
                keras_symbol=None,
                keras_adapter="prediction_mean",
                pytorch_adapter="prediction_mean",
                supported_tasks=["regression", "binary_classification", "multiclass_classification"],
                target_contract="continuous_same_shape",
                formula=r"\operatorname{PredictionMean}=\frac{1}{n}\sum_i\hat y_i",
            ),
            _mapped_option(
                "prediction_sum",
                "Prediction sum",
                "Sum of all values in the prediction tensor; target values are ignored.",
                keras_symbol=None,
                keras_adapter="prediction_sum",
                pytorch_adapter="prediction_sum",
                supported_tasks=["regression", "binary_classification", "multiclass_classification"],
                target_contract="continuous_same_shape",
                formula=r"\operatorname{PredictionSum}=\sum_i\hat y_i",
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

###############################################################################
class PersistentCompilationCatalogProvider:
    """Merge server-owned built-ins with installation-wide user definitions."""

    # -------------------------------------------------------------------------
    def __init__(self, uow_factory: Callable[[], Any]) -> None:
        self._uow_factory = uow_factory

    # -------------------------------------------------------------------------
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
