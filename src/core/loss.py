from abc import ABC, abstractmethod

from .types import Array


class LossFunction(ABC):
    """Contract for all loss (objective) functions.

    Implementors must provide both the scalar loss value and the gradient
    with respect to the **predictions** (not the parameters), which seeds
    the backward pass.
    """

    @abstractmethod
    def compute(self, y_true: Array, y_pred: Array) -> float:
        """Compute the scalar loss over a batch.

        Args:
            y_true: Ground-truth targets, shape ``(n_samples, n_outputs)``.
            y_pred: Model predictions, same shape as ``y_true``.

        Returns:
            Scalar loss value averaged over the batch.
        """
        ...

    @abstractmethod
    def gradient(self, y_true: Array, y_pred: Array) -> Array:
        """Compute ``∂L/∂y_pred``.

        This is the starting point for backpropagation.

        Args:
            y_true: Ground-truth targets, shape ``(n_samples, n_outputs)``.
            y_pred: Model predictions, same shape as ``y_true``.

        Returns:
            Gradient array with the same shape as ``y_pred``.
        """
        ...
