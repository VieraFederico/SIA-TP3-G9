from src.core.types import Array

from .trainer import BaseTrainer


class MiniBatchTrainer(BaseTrainer):
    """Mini-batch training: one update per chunk of ``batch_size`` samples.

    Shuffles data each epoch, then iterates over non-overlapping mini-batches.
    The epoch loss is the average over all mini-batch losses.

    Args:
        batch_size: Number of samples per mini-batch.  Defaults to 32.
        (All other args forwarded to ``BaseTrainer``.)
    """

    def __init__(self, *args, batch_size: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def train_epoch(self, X: Array, y: Array) -> float:
        """Iterate over mini-batches; return average mini-batch loss.

        Returns:
            Average loss over all mini-batches in the epoch.
        """
        raise NotImplementedError("TODO")
