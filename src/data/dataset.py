from __future__ import annotations

from dataclasses import dataclass

from src.core.types import Array


@dataclass
class Dataset:
    """Container for a feature matrix ``X`` and a target matrix ``y``.

    Optionally holds a human-readable ``name`` for logging.

    Attributes:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target matrix of shape ``(n_samples, n_outputs)``.
        name: Optional descriptive name (e.g. ``"train"``, ``"test"``).
    """

    X: Array
    y: Array
    name: str = ""

    def __len__(self) -> int:
        return len(self.X)

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X has {len(self.X)} samples but y has {len(self.y)}."
            )
