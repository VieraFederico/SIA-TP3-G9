import numpy as np
from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class AdamOptimizer(Optimizer):
    """Adam optimizer. Combina momentum y RMSprop con corrección de sesgo.

    m = β₁·m + (1-β₁)·g
    v = β₂·v + (1-β₂)·g²
    m̂ = m / (1 - β₁ᵗ),  v̂ = v / (1 - β₂ᵗ)
    Δw = -η · m̂ / (√v̂ + ε)
    """

    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._t: int = 0
        self._m: list[tuple[Array, Array]] = []
        self._v: list[tuple[Array, Array]] = []

    def update(
        self,
        params: list[tuple[Array, Array]],
        grads:  list[tuple[Array, Array]],
    ) -> list[tuple[Array, Array]]:
        """Aplica la actualización de Adam con corrección de sesgo."""
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        self._t = 0
        self._m = []
        self._v = []
