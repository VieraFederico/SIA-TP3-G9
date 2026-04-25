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




def plot_adaline_regression(X, y, model, title, output_path):
    """
    Plot ADALINE regression for 1-feature data.
    X shape: (n_samples, 1), y shape: (n_samples,)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten for plotting convenience
    x_vals = X[:, 0] if X.ndim == 2 else X
    y_vals = y

    # Sort x so fitted line is drawn cleanly
    order = np.argsort(x_vals)
    x_sorted = x_vals[order]
    X_sorted = x_sorted.reshape(-1, 1)
    y_pred_sorted = model.predict(X_sorted)

    plt.figure(figsize=(7, 5))
    plt.scatter(x_vals, y_vals, color="royalblue", edgecolor="black", s=45, label="Training data")
    plt.plot(x_sorted, y_pred_sorted, color="crimson", linewidth=2, label="ADALINE prediction")

    # Optional: show target relation if your dataset is y=2x
    # plt.plot(x_sorted, 2 * x_sorted, "k--", linewidth=1.5, label="Target y=2x")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_adaline_mse(errors_per_epoch, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(errors_per_epoch) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, errors_per_epoch, marker="o")
    plt.title("ADALINE Training Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Sum of squared errors (SSE)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()