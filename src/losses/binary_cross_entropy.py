from src.core.loss import LossFunction
from src.core.types import Array


class BinaryCrossEntropyLoss(LossFunction):
    """Binary Cross-Entropy loss for binary classification.

    L       = −(1/N) · Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]
    ∂L/∂ŷ  = (1/N) · (−y/ŷ + (1−y)/(1−ŷ))
    """

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """L = −(1/N) · Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]."""
        raise NotImplementedError("TODO")

    def gradient(self, y_true: Array, y_pred: Array) -> Array:
        """∂L/∂ŷ = (1/N) · (−y/ŷ + (1−y)/(1−ŷ))."""
        raise NotImplementedError("TODO")
