from __future__ import annotations

from pathlib import Path

from src.core.model import Model
from src.training.history import TrainingHistory


class ExperimentWriter:
    """Write all artefacts for a training run to ``experiments/<run_id>/``.

    Directory layout created:
        experiments/<run_id>/
            config.json       — snapshot of ExperimentConfig
            model.npz         — serialised model weights
            history.json      — per-epoch losses and times
            metrics.json      — final evaluation metrics

    Args:
        run_id: Unique identifier for this run (e.g. a timestamp string).
        base_dir: Root directory for experiments.  Defaults to
            ``Path("experiments")``.
    """

    def __init__(
        self, run_id: str, base_dir: Path = Path("experiments")
    ) -> None:
        self.run_id = run_id
        self.run_dir = base_dir / run_id

    def write(
        self,
        config,
        model: Model,
        history: TrainingHistory,
        metrics: dict[str, float],
    ) -> None:
        """Persist all artefacts to disk.

        Args:
            config: ``ExperimentConfig`` instance.
            model: Trained model.
            history: Training history.
            metrics: Final evaluation metrics dict.
        """
        raise NotImplementedError("TODO")


class ExperimentReader:
    """Read artefacts from a previously saved experiment run.

    Args:
        run_id: The run identifier to load.
        base_dir: Root directory for experiments.
    """

    def __init__(
        self, run_id: str, base_dir: Path = Path("experiments")
    ) -> None:
        self.run_id = run_id
        self.run_dir = base_dir / run_id

    def read_config(self):
        """Load and return the ``ExperimentConfig`` snapshot.

        Returns:
            ``ExperimentConfig`` deserialized from ``config.json``.
        """
        raise NotImplementedError("TODO")

    def read_history(self) -> TrainingHistory:
        """Load and return the ``TrainingHistory``.

        Returns:
            ``TrainingHistory`` deserialized from ``history.json``.
        """
        raise NotImplementedError("TODO")

    def read_metrics(self) -> dict[str, float]:
        """Load and return the final metrics dict.

        Returns:
            Dict from ``metrics.json``.
        """
        raise NotImplementedError("TODO")
