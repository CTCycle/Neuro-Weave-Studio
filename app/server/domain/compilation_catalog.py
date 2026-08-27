from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CompilationParameterType = Literal["number", "integer", "boolean", "enum"]

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
class CompilationOption(BaseModel):
    id: str
    label: str
    description: str
    parameters: list[CompilationParameter] = Field(default_factory=list)
    formula: str | None = None

###############################################################################
class CompilationCatalog(BaseModel):
    optimizers: list[CompilationOption] = Field(default_factory=list)
    losses: list[CompilationOption] = Field(default_factory=list)
    metrics: list[CompilationOption] = Field(default_factory=list)
    initializers: list[str] = Field(default_factory=list)
    regularizers: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


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
    constraints: dict[str, float] = {}
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
    return CompilationOption(id=identifier, label=label, description=description, parameters=parameters)


###############################################################################
def get_compilation_catalog() -> CompilationCatalog:
    return CompilationCatalog(
        optimizers=[
            _optimizer(
                "adam",
                "Adam",
                "Adaptive first-order optimizer with momentum estimates.",
                [
                    _number("learning_rate", "Learning rate", 0.001, "Step size for each update.", minimum=0.0),
                    _number("beta_1", "Beta 1", 0.9, "Exponential decay for the first moment.", minimum=0.0, maximum=1.0),
                    _number("beta_2", "Beta 2", 0.999, "Exponential decay for the second moment.", minimum=0.0, maximum=1.0),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                ],
            ),
            _optimizer(
                "adamw",
                "AdamW",
                "Adam with decoupled weight decay.",
                [
                    _number("learning_rate", "Learning rate", 0.001, "Step size for each update.", minimum=0.0),
                    _number("weight_decay", "Weight decay", 0.004, "Decoupled regularization coefficient.", minimum=0.0),
                    _number("beta_1", "Beta 1", 0.9, "Exponential decay for the first moment.", minimum=0.0, maximum=1.0),
                    _number("beta_2", "Beta 2", 0.999, "Exponential decay for the second moment.", minimum=0.0, maximum=1.0),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                ],
            ),
            _optimizer(
                "sgd",
                "SGD",
                "Stochastic gradient descent with optional momentum.",
                [
                    _number("learning_rate", "Learning rate", 0.01, "Step size for each update.", minimum=0.0),
                    _number("momentum", "Momentum", 0.0, "Momentum contribution from the previous update.", minimum=0.0, maximum=1.0),
                    CompilationParameter(id="nesterov", label="Nesterov", type="boolean", default=False, description="Use Nesterov momentum."),
                ],
            ),
            _optimizer(
                "rmsprop",
                "RMSprop",
                "Adaptive optimizer using a moving average of squared gradients.",
                [
                    _number("learning_rate", "Learning rate", 0.001, "Step size for each update.", minimum=0.0),
                    _number("rho", "Rho", 0.9, "Discount factor for the moving average.", minimum=0.0, maximum=1.0),
                    _number("momentum", "Momentum", 0.0, "Momentum contribution from the previous update.", minimum=0.0, maximum=1.0),
                    _number("epsilon", "Epsilon", 1e-7, "Numerical stability term.", minimum=0.0),
                    CompilationParameter(id="centered", label="Centered", type="boolean", default=False, description="Normalize gradients using their centered variance."),
                ],
            ),
        ],
        losses=[
            CompilationOption(id="mean_squared_error", label="Mean squared error", description="Squared difference between targets and predictions.", formula=r"\operatorname{MSE}=\frac{1}{n}\sum_i(y_i-\hat y_i)^2"),
            CompilationOption(id="mean_absolute_error", label="Mean absolute error", description="Absolute difference between targets and predictions.", formula=r"\operatorname{MAE}=\frac{1}{n}\sum_i\lvert y_i-\hat y_i\rvert"),
            CompilationOption(id="binary_crossentropy", label="Binary crossentropy", description="Crossentropy for binary targets.", formula=r"-y\log(\hat y)-(1-y)\log(1-\hat y)"),
            CompilationOption(id="categorical_crossentropy", label="Categorical crossentropy", description="Crossentropy for one-hot categorical targets.", formula=r"-\sum_c y_c\log(\hat y_c)"),
            CompilationOption(id="sparse_categorical_crossentropy", label="Sparse categorical crossentropy", description="Crossentropy for integer categorical targets.", formula=r"-\log(\hat y_{y})"),
            CompilationOption(id="huber", label="Huber", description="Quadratic near zero and linear for large residuals.", formula=r"L_\delta(a)=\begin{cases}\frac{1}{2}a^2 & \lvert a\rvert\leq\delta\\\delta(\lvert a\rvert-\frac{1}{2}\delta)&\text{otherwise}\end{cases}"),
            CompilationOption(id="kullback_leibler_divergence", label="Kullback-Leibler divergence", description="Divergence between target and predicted distributions.", formula=r"D_{KL}(P\parallel Q)=\sum_iP_i\log\frac{P_i}{Q_i}"),
        ],
        metrics=[
            CompilationOption(id="accuracy", label="Accuracy", description="Correct predictions divided by total predictions."),
            CompilationOption(id="binary_accuracy", label="Binary accuracy", description="Accuracy for binary predictions."),
            CompilationOption(id="categorical_accuracy", label="Categorical accuracy", description="Accuracy for one-hot categorical predictions."),
            CompilationOption(id="sparse_categorical_accuracy", label="Sparse categorical accuracy", description="Accuracy for integer categorical predictions."),
            CompilationOption(id="precision", label="Precision", description="Positive predictions that are correct.", formula=r"\frac{TP}{TP+FP}"),
            CompilationOption(id="recall", label="Recall", description="Actual positives that are identified.", formula=r"\frac{TP}{TP+FN}"),
            CompilationOption(id="auc", label="AUC", description="Area under the receiver operating characteristic curve."),
            CompilationOption(id="mean_absolute_error", label="Mean absolute error", description="Mean absolute prediction error."),
            CompilationOption(id="mean_squared_error", label="Mean squared error", description="Mean squared prediction error."),
            CompilationOption(id="root_mean_squared_error", label="Root mean squared error", description="Square root of mean squared prediction error."),
        ],
        initializers=["uniform", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal", "zeros", "ones", "random_normal", "random_uniform", "orthogonal"],
        regularizers=["none", "l1", "l2", "l1_l2"],
        constraints=["none", "max_norm", "min_max_norm", "non_neg", "unit_norm"],
    )
