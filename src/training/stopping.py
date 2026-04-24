from __future__ import annotations

from abc import ABC, abstractmethod

from .history import TrainingHistory


class StoppingCriterion(ABC):
    """Contract for early-stopping / termination conditions.

    Multiple criteria can be combined with ``CompositeStopping`` (logical
    OR: stops when *any* criterion fires).
    """

    @abstractmethod
    def should_stop(self, history: TrainingHistory) -> bool:
        """Return ``True`` if training should halt.

        Args:
            history: Training history collected so far.
        """
        ...


class MaxEpochs(StoppingCriterion):
    """Stop after a fixed number of epochs.

    Args:
        max_epochs: Maximum number of training epochs.
    """

    def __init__(self, max_epochs: int) -> None:
        self.max_epochs = max_epochs

    def should_stop(self, history: TrainingHistory) -> bool:
        """Return True when len(train_losses) >= max_epochs."""
        raise NotImplementedError("TODO")


class LossThreshold(StoppingCriterion):
    """Stop when training loss falls below a threshold.

    Args:
        threshold: Target loss value.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def should_stop(self, history: TrainingHistory) -> bool:
        """Return True when the last training loss <= threshold."""
        raise NotImplementedError("TODO")


class EarlyStopping(StoppingCriterion):
    """Stop when validation loss has not improved for ``patience`` epochs.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum decrease in val loss to count as an improvement.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._epochs_no_improve: int = 0

    def should_stop(self, history: TrainingHistory) -> bool:
        """Return True when val_loss hasn't improved for ``patience`` epochs."""
        raise NotImplementedError("TODO")


class CompositeStopping(StoppingCriterion):
    """Logical OR of multiple stopping criteria.

    Stops training as soon as *any* of the provided criteria fires.

    Args:
        criteria: One or more ``StoppingCriterion`` instances.
    """

    def __init__(self, *criteria: StoppingCriterion) -> None:
        self.criteria = list(criteria)

    def should_stop(self, history: TrainingHistory) -> bool:
        """Return True if any criterion returns True."""
        raise NotImplementedError("TODO")
