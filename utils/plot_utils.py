from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def plot_decision_boundary(X, y, model, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    for label in np.unique(y):
        pts = X[y == label]
        plt.scatter(pts[:, 0], pts[:, 1],
                    s=100, edgecolor='black',
                    label=f"Class {label}")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    # replace plt.show()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_adaline_regression(
        X, y, model, title, output_path,
        xlim=(-6, 6), ylim=(-8, 18), centered=True
):
    """
    Plot ADALINE regression for 1-feature data.
    X shape: (n_samples, 1) or (n_samples,)
    y shape: (n_samples,)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_vals = X[:, 0] if np.ndim(X) == 2 else X
    y_vals = y

    order = np.argsort(x_vals)
    x_sorted = x_vals[order]
    X_sorted = x_sorted.reshape(-1, 1)
    y_pred_sorted = model.predict(X_sorted)

    plt.figure(figsize=(7, 5))
    plt.scatter(x_vals, y_vals, color="royalblue", edgecolor="black", s=45, label="Training data")
    plt.plot(x_sorted, y_pred_sorted, color="crimson", linewidth=2, label="ADALINE prediction")

    # Fixed scale
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    # Keep visual center stable
    if centered:
        plt.axhline(0, color="gray", linewidth=1, alpha=0.6)
        plt.axvline(0, color="gray", linewidth=1, alpha=0.6)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_adaline_mse(error_series, output_path, title="ADALINE Training Error Comparison"):
    """
    error_series: dict[str, list[float]] or list[tuple[str, list[float]]]
    Example:
        {
            "eta=0.01": [..],
            "eta=0.05": [..]
        }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))

    if isinstance(error_series, dict):
        items = error_series.items()
    else:
        items = error_series  # list of (label, errors)

    for label, errors in items:
        epochs = np.arange(1, len(errors) + 1)
        plt.plot(epochs, errors, marker="o", label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Sum of squared errors (SSE)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_error_scatter(results, output_path):
    labels = list(results.keys())

    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        y = results[label]
        x = np.full(len(y), i, dtype=float)
        jitter = (np.random.rand(len(y)) - 0.5) * 0.2  # small horizontal jitter
        plt.scatter(x + jitter, y, alpha=0.7)

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title("Final Error Scatter by Hyperparameter Combo")
    plt.ylabel("Final MSE (last epoch)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
