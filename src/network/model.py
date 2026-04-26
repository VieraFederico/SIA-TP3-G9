# src/network/model.py  ← archivo nuevo
from typing import Protocol
import numpy as np

class Model(Protocol):
    """Cualquier cosa que tenga forward y backward es un Model."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        ...