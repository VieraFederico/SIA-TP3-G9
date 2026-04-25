from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class MomentumOptimizer(Optimizer):
    """Gradient descent con momentum. v = β·v - η·∂E/∂w,  Δw = v"""

    def __init__(self, eta: float, beta: float = 0.9):
        self.eta = eta
        self.beta = beta
        self._velocities: list[Array] = []

    def update(self, weights: list[Array], grads: list[Array]) -> list[Array]:
        """v = β·v - η·grad,  w = w + v"""
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        """Reinicia las velocidades acumuladas."""
        raise NotImplementedError("TODO")
