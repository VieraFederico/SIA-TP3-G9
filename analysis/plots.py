"""Matplotlib helpers for offline analysis — only numpy arrays and ``RunHistory``, no live models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def plot_loss_curve(
    history: Any,
    *,
    save_to: str | Path | None = None,
    title: str = "Training loss",
) -> None:
    """Plot train (and optional val) loss vs epoch."""
    if hasattr(history, "train_losses"):
        train = list(history.train_losses)  # type: ignore[union-attr]
        val = list(history.val_losses)  # type: ignore[union-attr]
    else:
        d = dict(history)
        train = list(d.get("train_losses", []))
        val = list(d.get("val_losses", []))

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train) + 1)
    ax.plot(epochs, train, label="train")
    if val:
        ve = range(1, len(val) + 1)
        ax.plot(ve, val, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    labels: list[str] | None = None,
    save_to: str | Path | None = None,
    title: str = "Confusion matrix",
) -> None:
    """Plot confusion matrix for integer class labels (classification)."""
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    n_class = int(max(yt.max(), yp.max(), 0)) + 1
    cm = np.zeros((n_class, n_class), dtype=float)
    for t, p in zip(yt, yp):
        cm[t, p] += 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    if labels:
        ax.set_xticks(range(n_class))
        ax.set_yticks(range(n_class))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(range(n_class))
        ax.set_yticks(range(n_class))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)


def plot_decision_boundary(
    *args: Any,
    **kwargs: Any,
) -> None:
    """Not supported in the isolated ``analysis/`` package (requires a loaded ``Model``).

    Run from a notebook with access to ``src`` if you need decision boundaries, or
    save predictions to disk and plot the class regions from 2D features only.

    Raises:
        NotImplementedError: Always — use a notebook bridge for live models.
    """
    raise NotImplementedError(
        "plot_decision_boundary is not implemented in analysis/: "
        "it requires a Model instance; keep analysis limited to disk artefacts."
    )


def plot_metric_comparison(
    runs_df: Any,
    metric: str,
    *,
    save_to: str | Path | None = None,
    title: str | None = None,
) -> None:
    """Bar chart of one metric across runs (``run_id`` vs ``metric`` column)."""
    import pandas as pd

    if not isinstance(runs_df, pd.DataFrame):
        raise TypeError("runs_df must be a pandas DataFrame (e.g. from compare_runs)")

    if metric not in runs_df.columns:
        raise KeyError(f"metric column {metric!r} not in DataFrame")

    fig, ax = plt.subplots(figsize=(max(6, len(runs_df) * 0.5), 4))
    xs = range(len(runs_df))
    ax.bar(xs, runs_df[metric].tolist())
    ax.set_xticks(list(xs))
    ax.set_xticklabels(runs_df["run_id"].tolist(), rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by run")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)
