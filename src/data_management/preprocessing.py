from src.activation.activation import Array


def normalize(X: Array) -> Array:
    """Escala cada feature a [0, 1]: x' = (x - min) / (max - min)"""
    raise NotImplementedError("TODO")


def standardize(X: Array) -> Array:
    """Estandariza cada feature: x' = (x - μ) / σ"""
    raise NotImplementedError("TODO")


def one_hot_encode(zeta: Array, n_classes: int) -> Array:
    """Convierte etiquetas enteras en vectores one-hot de longitud n_classes."""
    raise NotImplementedError("TODO")
