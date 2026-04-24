from src.core.types import Array

from .trainer import BaseTrainer


class BatchTrainer(BaseTrainer):
    """Full-batch training: one weight update per epoch using all samples.

    Computes the gradient over the entire dataset before updating
    parameters.  Slower per update but produces the true gradient.
    """

    def train_epoch(self, X: Array, y: Array) -> float:
        """Forward + backward over the full dataset; one optimizer step.

        Returns:
            Scalar loss for the epoch.
        """
        raise NotImplementedError("TODO")
