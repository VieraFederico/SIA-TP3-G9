from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

if TYPE_CHECKING:
    import pandas as pd
    from src.network.multilayer_perceptron import MultilayerPerceptron


def plot_error_curve(history: dict, save_to: str | None = None) -> None:
    """Grafica E vs épocas (train y val)."""
    raise NotImplementedError("TODO")


def plot_confusion_matrix(zeta: Array, O: Array, save_to: str | None = None) -> None:
    """Grafica la matriz de confusión de las predicciones."""
    raise NotImplementedError("TODO")


def plot_decision_boundary(
    model: MultilayerPerceptron,
    X: Array,
    zeta: Array,
    save_to: str | None = None,
) -> None:
    """Grafica la frontera de decisión del modelo sobre el espacio de entrada 2D."""
    raise NotImplementedError("TODO")


def plot_metric_comparison(df: pd.DataFrame, metric: str, save_to: str | None = None) -> None:
    """Grafica comparación de una métrica entre múltiples runs."""
    raise NotImplementedError("TODO")
