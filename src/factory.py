"""Factory module — translates ExperimentConfig strings into concrete objects.

Adding a new activation / optimizer / loss / metric requires:
    1. Implementing the class in the appropriate subpackage.
    2. Registering it in the corresponding dict below.
    3. Using its string key in a JSON config.

No other file needs to change.
"""

from __future__ import annotations

from src.activations.identity import IdentityActivation
from src.activations.logistic import LogisticActivation
from src.activations.relu import ReLUActivation
from src.activations.step import StepActivation
from src.activations.tanh import TanhActivation
from src.config import ExperimentConfig
from src.core.activation import ActivationFunction
from src.core.loss import LossFunction
from src.core.model import Model
from src.core.optimizer import Optimizer
from src.data.dataset import Dataset
from src.losses.binary_cross_entropy import BinaryCrossEntropyLoss
from src.losses.categorical_cross_entropy import CategoricalCrossEntropyLoss
from src.losses.mse import MSELoss
from src.metrics.classification import (
    AccuracyMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)
from src.metrics.metric import Metric
from src.metrics.regression import MAEMetric, MSEMetric, R2Metric
from src.models.mlp import MultilayerPerceptron
from src.models.simple_perceptron import SimplePerceptron
from src.optimizers.adam import AdamOptimizer
from src.optimizers.gradient_descent import GradientDescentOptimizer
from src.optimizers.momentum import MomentumOptimizer
from src.training.batch_trainer import BatchTrainer
from src.training.minibatch_trainer import MiniBatchTrainer
from src.training.online_trainer import OnlineTrainer
from src.training.stopping import (
    CompositeStopping,
    EarlyStopping,
    LossThreshold,
    MaxEpochs,
    StoppingCriterion,
)
from src.training.trainer import BaseTrainer

# Registry dicts — the single place that maps config strings to classes.
ACTIVATIONS: dict[str, type[ActivationFunction]] = {
    "step": StepActivation,
    "identity": IdentityActivation,
    "tanh": TanhActivation,
    "logistic": LogisticActivation,
    "relu": ReLUActivation,
}

LOSSES: dict[str, type[LossFunction]] = {
    "mse": MSELoss,
    "binary_cross_entropy": BinaryCrossEntropyLoss,
    "categorical_cross_entropy": CategoricalCrossEntropyLoss,
}

OPTIMIZERS: dict[str, type[Optimizer]] = {
    "gradient_descent": GradientDescentOptimizer,
    "momentum": MomentumOptimizer,
    "adam": AdamOptimizer,
}

TRAINERS: dict[str, type[BaseTrainer]] = {
    "online": OnlineTrainer,
    "batch": BatchTrainer,
    "minibatch": MiniBatchTrainer,
}

METRICS: dict[str, type[Metric]] = {
    "mse": MSEMetric,
    "mae": MAEMetric,
    "r2": R2Metric,
    "accuracy": AccuracyMetric,
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "f1": F1Metric,
}

# Weight-init scheme names forwarded to models/layers (see ``cfg.initializer``).
INITIALIZERS: dict[str, str] = {
    "uniform": "uniform",
    "xavier": "xavier",
    "he": "he",
}

STOPPING_CRITERIA: dict[str, type[StoppingCriterion]] = {
    "max_epochs": MaxEpochs,
    "loss_threshold": LossThreshold,
    "early_stopping": EarlyStopping,
    "composite": CompositeStopping,
}


def build_activation(cfg: ExperimentConfig) -> ActivationFunction:
    """Instantiate the activation from ``cfg.activation`` and ``cfg.activation_beta``.

    Args:
        cfg: Experiment configuration.

    Returns:
        Concrete ``ActivationFunction`` instance.

    Raises:
        KeyError: If ``cfg.activation`` is not in ``ACTIVATIONS``.
    """
    raise NotImplementedError("TODO")


def build_optimizer(cfg: ExperimentConfig) -> Optimizer:
    """Instantiate the optimizer from ``cfg.optimizer``, ``cfg.learning_rate``,
    and ``cfg.optimizer_params``.

    Args:
        cfg: Experiment configuration.

    Returns:
        Concrete ``Optimizer`` instance.

    Raises:
        KeyError: If ``cfg.optimizer`` is not in ``OPTIMIZERS``.
    """
    raise NotImplementedError("TODO")


def build_loss(cfg: ExperimentConfig) -> LossFunction:
    """Instantiate the loss function from ``cfg.loss``.

    Args:
        cfg: Experiment configuration.

    Returns:
        Concrete ``LossFunction`` instance.

    Raises:
        KeyError: If ``cfg.loss`` is not in ``LOSSES``.
    """
    raise NotImplementedError("TODO")


def build_metrics(cfg: ExperimentConfig) -> list[Metric]:
    """Instantiate all metrics listed in ``cfg.metrics``.

    Args:
        cfg: Experiment configuration.

    Returns:
        List of concrete ``Metric`` instances.

    Raises:
        KeyError: If any metric name is not in ``METRICS``.
    """
    raise NotImplementedError("TODO")


def build_model(cfg: ExperimentConfig) -> Model:
    """Instantiate the model described by ``cfg.model_type`` and
    ``cfg.architecture``.

    Args:
        cfg: Experiment configuration.

    Returns:
        Uninitialised (randomly-weighted) concrete ``Model`` instance.

    Raises:
        KeyError: If ``cfg.model_type`` is unknown.
    """
    raise NotImplementedError("TODO")


def build_trainer(
    cfg: ExperimentConfig,
    model: Model,
) -> BaseTrainer:
    """Assemble a trainer from all components described in ``cfg``.

    Builds and wires: optimizer, loss, stopping criterion (composite),
    metrics, logger, and trainer.

    Args:
        cfg: Experiment configuration.
        model: Already-built model to train.

    Returns:
        Configured concrete ``BaseTrainer`` ready to call ``.fit()``.
    """
    raise NotImplementedError("TODO")


def build_experiment(cfg: ExperimentConfig) -> tuple[Model, BaseTrainer, object]:
    """Top-level factory: build everything from a config.

    Args:
        cfg: Experiment configuration.

    Returns:
        ``(model, trainer, split_datasets)`` where ``split_datasets`` is an
        object with ``.train``, ``.val``, and ``.test`` ``Dataset``
        attributes.
    """
    raise NotImplementedError("TODO")
