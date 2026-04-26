from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

Array = NDArray[np.float64]

if TYPE_CHECKING:
    import pandas as pd


def plot_error_curve(history: dict, output_path: str) -> None:
    """Grafica E vs épocas (train y val).

    history — dict con "train_error" y "val_error", lo que devuelve trainer.fit()

    Si val_error sube mientras train_error baja → overfitting.
    Si ambos bajan juntos → el modelo está aprendiendo bien.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_error"]) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_error"], marker="o", label="Train error")
    plt.plot(epochs, history["val_error"],   marker="o", label="Val error", linestyle="--")
    plt.title("Error por época")
    plt.xlabel("Época")
    plt.ylabel("E")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(zeta: Array, O: Array, save_to: str | None = None) -> None:
    """Grafica la matriz de confusión de las predicciones."""
    raise NotImplementedError("TODO")


def plot_decision_boundary(X: Array, zeta: Array, model, title: str, output_path: str) -> None:
    """Grafica la frontera de decisión de un clasificador binario.

    Usado para el perceptrón escalón (ej: compuerta AND).
    Crea una grilla de puntos, predice la clase de cada uno,
    y colorea el fondo según la predicción — la frontera visible
    es exactamente el hiperplano w₁x₁ + w₂x₂ + w₀ = 0.

    X     — matriz (n, 2) con las dos features
    zeta  — vector (n,) con las etiquetas reales {0, 1}
    model — cualquier objeto con forward(x) — Neuron, MLP, etc.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rango del gráfico: un poco más grande que los datos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Grilla de 300x300 puntos que cubre todo el espacio visible
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # c_[...] apila las coordenadas en columnas: cada fila es un punto (x1, x2)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model.forward(point) for point in grid_points])
    Z = Z.reshape(xx.shape)

    # contourf colorea las regiones — la frontera aparece donde cambia el color
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    for label in np.unique(zeta):
        puntos = X[zeta == label]
        plt.scatter(puntos[:, 0], puntos[:, 1],
                    s=100, edgecolor="black", label=f"Clase {label}")

    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regression(X: Array, zeta: Array, model, title: str, output_path: str,
                    xlim=(-6, 6), ylim=(-1.5, 1.5)) -> None:
    """Grafica la curva predicha por el modelo vs los datos reales.

    Usado para perceptrón lineal y tanh.
    Equivale a plot_adaline_regression del tutorial.

    X     — array (n, 1) o (n,) con la feature
    zeta  — array (n,) con los valores reales
    model — cualquier objeto con forward(x)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aplanamos X para graficar en 2D
    x_vals = X[:, 0] if X.ndim == 2 else X

    # Ordenamos para que la línea se dibuje de izquierda a derecha
    orden    = np.argsort(x_vals)
    x_sorted = x_vals[orden]
    X_sorted = x_sorted.reshape(-1, 1)

    # Predecimos sobre los puntos ordenados para dibujar la curva
    y_pred = np.array([model.forward(xi) for xi in X_sorted])

    plt.figure(figsize=(7, 5))
    plt.scatter(x_vals, zeta,
                color="royalblue", edgecolor="black", s=45, label="Datos reales")
    plt.plot(x_sorted, y_pred,
             color="crimson", linewidth=2, label="Predicción del modelo")

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.axhline(0, color="gray", linewidth=1, alpha=0.6)
    plt.axvline(0, color="gray", linewidth=1, alpha=0.6)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(df: pd.DataFrame, metric: str, save_to: str | None = None) -> None:
    """Grafica comparación de una métrica entre múltiples runs."""
    raise NotImplementedError("TODO")
