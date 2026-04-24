from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """Record of per-epoch training and validation metrics.

    Attributes:
        train_losses: Loss value after each training epoch.
        val_losses: Loss value after each validation epoch (may be empty if
            no validation set was provided).
        epoch_times: Wall-clock time in seconds for each epoch.
        extra_metrics: Optional dict mapping metric names to per-epoch lists
            (e.g. ``{"accuracy": [0.5, 0.6, …]}``).
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    extra_metrics: dict[str, list[float]] = field(default_factory=dict)

    def append_train_loss(self, loss: float) -> None:
        """Append a training loss value for the current epoch."""
        self.train_losses.append(loss)

    def append_val_loss(self, loss: float) -> None:
        """Append a validation loss value for the current epoch."""
        self.val_losses.append(loss)

    def append_epoch_time(self, elapsed: float) -> None:
        """Append a wall-clock epoch duration in seconds."""
        self.epoch_times.append(elapsed)

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epoch_times": self.epoch_times,
            "extra_metrics": self.extra_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainingHistory:
        """Deserialise from a plain dict (as loaded from JSON)."""
        return cls(
            train_losses=data.get("train_losses", []),
            val_losses=data.get("val_losses", []),
            epoch_times=data.get("epoch_times", []),
            extra_metrics=data.get("extra_metrics", {}),
        )
