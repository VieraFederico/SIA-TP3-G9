from __future__ import annotations

from pathlib import Path

from src.core.model import Model


def save_model(model: Model, path: Path) -> None:
    """Persist model parameters to a ``.npz`` file.

    A JSON sidecar ``<path>.json`` is written alongside the ``.npz`` with
    the model class name and constructor arguments, so the model can be
    fully reconstructed by ``load_model``.

    Args:
        model: Trained model to save.
        path: Destination path (e.g. ``experiments/run_01/model.npz``).
    """
    raise NotImplementedError("TODO")


def load_model(path: Path) -> Model:
    """Restore a model from a ``.npz`` file and its JSON sidecar.

    Args:
        path: Path to the ``.npz`` file written by ``save_model``.

    Returns:
        Model instance with weights restored.

    Raises:
        FileNotFoundError: If ``path`` or the sidecar JSON is missing.
    """
    raise NotImplementedError("TODO")
