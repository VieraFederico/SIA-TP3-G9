from abc import ABC, abstractmethod

from src.core.types import Array


class Metric(ABC):
    """Contract for evaluation metrics.

    Implementations must be stateless — ``compute`` does not modify ``self``.
    """

    @abstractmethod
    def name(self) -> str:
        """Return a short, unique string identifier for this metric.

        Used as the key in the results dict returned by ``Trainer.evaluate``.
        """
        ...

    @abstractmethod
    def compute(self, y_true: Array, y_pred: Array) -> float:
        """Compute the metric and return a scalar.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions (raw outputs, not class indices).

        Returns:
            Scalar metric value.
        """
        ...
