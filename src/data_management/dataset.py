from dataclasses import dataclass
from src.activation.activation import Array


@dataclass
class Dataset:
    """Contiene X (entradas) y zeta (salidas esperadas ζ)."""

    X: Array
    zeta: Array
