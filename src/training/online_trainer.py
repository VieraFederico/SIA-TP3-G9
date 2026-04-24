from src.core.types import Array

from .trainer import BaseTrainer


class OnlineTrainer(BaseTrainer):
    """Online (stochastic) training: one weight update per sample.

    Each epoch shuffles the dataset and performs ``n_samples`` gradient
    updates, one sample at a time.  This is the classic Rosenblatt /
    ADALINE training mode and is equivalent to SGD with batch_size=1.
    """

    def train_epoch(self, X: Array, y: Array) -> float:
        """Iterate over shuffled samples, update once per sample.

        Returns:
            Average loss over all samples in the epoch.
        """
        raise NotImplementedError("TODO")
