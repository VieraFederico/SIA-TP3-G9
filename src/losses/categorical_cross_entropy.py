from src.core.loss import LossFunction
from src.core.types import Array


class CategoricalCrossEntropyLoss(LossFunction):
    """Categorical Cross-Entropy loss for multi-class classification.

    Expects ``y_true`` to be one-hot encoded and ``y_pred`` to be softmax
    probabilities.

    L       = −(1/N) · Σ_i Σ_k y_ik · log(ŷ_ik)
    ∂L/∂ŷ  = −(1/N) · y / ŷ  (element-wise)
    """

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """L = −(1/N) · Σ y · log(ŷ)."""
        raise NotImplementedError("TODO")

    def gradient(self, y_true: Array, y_pred: Array) -> Array:
        """∂L/∂ŷ = −(1/N) · y / ŷ."""
        raise NotImplementedError("TODO")
