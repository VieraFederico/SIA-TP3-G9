from src.core.loss import LossFunction
from src.core.types import Array


class MSELoss(LossFunction):
    """Mean Squared Error loss.

    L       = (1/N) · Σ (y_pred − y_true)²
    ∂L/∂ŷ  = (2/N) · (y_pred − y_true)
    """

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """L = (1/N) · Σ (y_pred − y_true)²."""
        raise NotImplementedError("TODO")

    def gradient(self, y_true: Array, y_pred: Array) -> Array:
        """∂L/∂ŷ = (2/N) · (y_pred − y_true)."""
        raise NotImplementedError("TODO")
