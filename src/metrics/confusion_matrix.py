from __future__ import annotations

from src.core.types import Array


class ConfusionMatrix:
    """Compute and store a confusion matrix for classification results.

    Args:
        n_classes: Number of classes.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.matrix: Array | None = None

    def compute(self, y_true: Array, y_pred: Array) -> Array:
        """Build the n_classes × n_classes confusion matrix.

        Args:
            y_true: True class indices, shape ``(n_samples,)``.
            y_pred: Predicted class indices, shape ``(n_samples,)``.

        Returns:
            Confusion matrix of shape ``(n_classes, n_classes)``.
            ``matrix[i, j]`` = number of samples of class ``i`` predicted
            as class ``j``.
        """
        raise NotImplementedError("TODO")

    def as_array(self) -> Array:
        """Return the stored matrix.  Call ``compute`` first."""
        raise NotImplementedError("TODO")
