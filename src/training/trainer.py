from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.core.loss import LossFunction
from src.core.model import Model
from src.core.optimizer import Optimizer
from src.core.types import Array

from .history import TrainingHistory
from .stopping import StoppingCriterion

if TYPE_CHECKING:
    from src.metrics.metric import Metric
    from src.utils.logger import Logger


class Trainer(ABC):
    """Abstract trainer — defines the contract for all training strategies.

    The only abstract method is ``train_epoch``, which encapsulates the
    difference between *online*, *batch*, and *mini-batch* training.

    Concrete subclasses must implement ``train_epoch``; all other logic
    (the training loop, evaluation, history recording) lives in
    ``BaseTrainer``.
    """

    @abstractmethod
    def train_epoch(self, X: Array, y: Array) -> float:
        """Run one full pass over the training data and return the epoch loss.

        Args:
            X: Training inputs, shape ``(n_samples, n_features)``.
            y: Training targets, shape ``(n_samples, n_outputs)``.

        Returns:
            Scalar loss averaged over the epoch.
        """
        ...


class BaseTrainer(Trainer, ABC):
    """Provides the full training loop on top of the abstract ``train_epoch``.

    Args:
        model: The model to train (any ``Model`` subclass).
        loss: Loss function used for both training and evaluation.
        optimizer: Parameter update rule.
        stopping: Criterion (or composite) that decides when to stop.
        metrics: Optional list of additional metrics to track per epoch.
        logger: Logger instance for progress output.
    """

    def __init__(
        self,
        model: Model,
        loss: LossFunction,
        optimizer: Optimizer,
        stopping: StoppingCriterion,
        metrics: list[Metric] | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.stopping = stopping
        self.metrics = metrics or []
        self.logger = logger

    def fit(
        self,
        X_train: Array,
        y_train: Array,
        X_val: Array | None = None,
        y_val: Array | None = None,
    ) -> TrainingHistory:
        """Run the full training loop until the stopping criterion fires.

        Args:
            X_train: Training inputs.
            y_train: Training targets.
            X_val: Optional validation inputs.
            y_val: Optional validation targets.

        Returns:
            ``TrainingHistory`` populated with per-epoch losses and times.
        """
        raise NotImplementedError("TODO")

    def evaluate(self, X: Array, y: Array) -> dict[str, float]:
        """Compute loss + all configured metrics on a dataset.

        Args:
            X: Input matrix.
            y: Target matrix.

        Returns:
            Dict mapping metric names to scalar values, e.g.
            ``{"loss": 0.05, "accuracy": 0.97}``.
        """
        raise NotImplementedError("TODO")
