import numpy as np
from src.activation.activation import Array


def normalize(X: Array) -> Array:
    """Escala cada feature a [0, 1]: x' = (x - min) / (max - min)"""
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    return (X - Xmin) / (Xmax - Xmin)


def normalize_with_params(X: Array, Xmin: Array, Xmax: Array) -> Array:
    """Aplica normalización usando parámetros pre-calculados (para val/test)."""
    return (X - Xmin) / (Xmax - Xmin)


def standardize(X: Array) -> Array:
    """Estandariza cada feature: x' = (x - μ) / σ. Columnas constantes quedan en 0."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    return (X - mean) / safe_std


def standardize_with_params(X: Array, mean: Array, std: Array) -> Array:
    """Aplica estandarización usando parámetros pre-calculados (para val/test)."""
    safe_std = np.where(std == 0, 1.0, std)
    return (X - mean) / safe_std


def one_hot_encode(zeta: Array, n_classes: int) -> Array:
    """Convierte etiquetas enteras en vectores one-hot de longitud n_classes."""
    result = np.zeros((len(zeta), n_classes))
    result[np.arange(len(zeta)), zeta.astype(int)] = 1.0
    return result
