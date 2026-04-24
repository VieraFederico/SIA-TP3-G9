from __future__ import annotations

import sys


class Logger:
    """Lightweight training-progress logger.

    Writes structured messages to ``stdout`` (or a provided stream).
    All print statements in the codebase must go through a ``Logger``
    instance — never use bare ``print()``.

    Args:
        name: Logger name shown in output (e.g. experiment name).
        verbose: If ``False``, suppress per-epoch output.
        stream: Output stream.  Defaults to ``sys.stdout``.
    """

    def __init__(
        self,
        name: str = "experiment",
        verbose: bool = True,
        stream=None,
    ) -> None:
        self.name = name
        self.verbose = verbose
        self._stream = stream or sys.stdout

    def info(self, message: str) -> None:
        """Write an informational message."""
        raise NotImplementedError("TODO")

    def epoch(self, epoch: int, total: int, loss: float, **extra: float) -> None:
        """Write a per-epoch progress line.

        Args:
            epoch: Current epoch number (1-based).
            total: Total number of epochs.
            loss: Training loss for this epoch.
            **extra: Additional named metric values to append.
        """
        raise NotImplementedError("TODO")

    def warning(self, message: str) -> None:
        """Write a warning message."""
        raise NotImplementedError("TODO")
