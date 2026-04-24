from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Single source of truth for an experiment.

    Every field maps 1-to-1 to a JSON key.  The factory reads these strings
    and constructs the concrete objects (model, optimizer, …).

    Attributes:
        name: Human-readable experiment identifier.
        seed: Global random seed for full reproducibility.

        data_path: Path to the CSV dataset file.
        target_column: Name of the CSV column used as target.
        preprocessing: Ordered list of preprocessor names to apply.
            Supported: ``"standard_scaler"``, ``"normalize"``,
            ``"one_hot"``.
        split: Split configuration dict.  Must contain ``"type"``
            (``"hold_out"`` or ``"k_fold"``).
            For ``hold_out``: ``{"type": "hold_out", "train": 0.7,
            "val": 0.15, "test": 0.15}``.
            For ``k_fold``: ``{"type": "k_fold", "k": 5}``.

        model_type: ``"simple_perceptron"`` or ``"mlp"``.
        architecture: Layer sizes ``[n_inputs, …, n_outputs]``.
        activation: Activation function name for hidden + output layers
            (Simple Perceptron uses one; MLP uses hidden vs output split
            handled by the factory).
        activation_beta: β parameter for tanh / logistic activations.
        initializer: Weight init scheme — ``"uniform"``, ``"xavier"``,
            or ``"he"``.

        trainer: Training mode — ``"online"``, ``"batch"``, or
            ``"minibatch"``.
        batch_size: Mini-batch size (ignored for online / batch trainers).
        loss: Loss function name — ``"mse"``, ``"binary_cross_entropy"``,
            or ``"categorical_cross_entropy"``.
        optimizer: Optimizer name — ``"gradient_descent"``,
            ``"momentum"``, or ``"adam"``.
        learning_rate: Step size η.
        optimizer_params: Extra keyword arguments forwarded to the
            optimizer constructor (e.g. ``{"beta": 0.9}`` for Momentum,
            ``{"beta1": 0.9, "beta2": 0.999}`` for Adam).

        max_epochs: Hard epoch ceiling (always active).
        loss_threshold: Stop when train loss ≤ this value.  ``None``
            to disable.
        early_stopping_patience: Stop when val loss hasn't improved for
            this many epochs.  ``None`` to disable.

        metrics: List of metric names to compute and log each epoch.
    """

    # Identification
    name: str
    seed: int

    # Data
    data_path: str
    target_column: str
    preprocessing: list[str]
    split: dict

    # Model
    model_type: str
    architecture: list[int]
    activation: str
    activation_beta: float = 1.0
    initializer: str = "xavier"

    # Training
    trainer: str = "minibatch"
    batch_size: int = 32
    loss: str = "mse"
    optimizer: str = "gradient_descent"
    learning_rate: float = 0.01
    optimizer_params: dict = field(default_factory=dict)

    # Stopping criteria
    max_epochs: int = 1000
    loss_threshold: float | None = 0.001
    early_stopping_patience: int | None = None

    # Metrics
    metrics: list[str] = field(default_factory=lambda: ["mse", "accuracy"])


def load_config(path: Path | str) -> ExperimentConfig:
    """Deserialise an ``ExperimentConfig`` from a JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Populated ``ExperimentConfig`` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required fields are missing from the JSON.
    """
    raise NotImplementedError("TODO")


def save_config(cfg: ExperimentConfig, path: Path | str) -> None:
    """Serialise an ``ExperimentConfig`` to a JSON file.

    The output is pretty-printed (indent=2) and stable (sorted keys).

    Args:
        cfg: Config to save.
        path: Destination file path.
    """
    raise NotImplementedError("TODO")
