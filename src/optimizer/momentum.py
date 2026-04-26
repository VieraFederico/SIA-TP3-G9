from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class MomentumOptimizer(Optimizer):
    """Gradient descent con momentum. v = β·v - η·∂E/∂w,  Δw = v"""

    def __init__(self, learning_rate: float, beta: float = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self._velocities: list[Array] = []

    def update(self, weights: list[Array], grads: list[Array]) -> list[Array]:
        """v = β·v - η·grad,  w = w + v"""
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        """Reinicia las velocidades acumuladas."""
        raise NotImplementedError("TODO")
