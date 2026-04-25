from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class GradientDescent(Optimizer):
    """Descenso de gradiente estándar. Δw = -η · ∂E/∂w"""

    def __init__(self, eta: float):
        self.eta = eta

    def update(self, weights: list[Array], grads: list[Array]) -> list[Array]:
        """Δw = -η · grad"""
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        """Sin estado interno."""
        raise NotImplementedError("TODO")
