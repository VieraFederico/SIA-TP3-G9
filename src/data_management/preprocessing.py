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
    """Estandariza cada feature: x' = (x - μ) / σ"""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def standardize_with_params(X: Array, mean: Array, std: Array) -> Array:
    """Aplica estandarización usando parámetros pre-calculados (para val/test)."""
    return (X - mean) / std


def one_hot_encode(zeta: Array, n_classes: int) -> Array:
    """Convierte etiquetas enteras en vectores one-hot de longitud n_classes."""
    raise NotImplementedError("TODO")
