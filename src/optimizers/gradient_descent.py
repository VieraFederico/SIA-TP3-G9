from src.core.optimizer import Optimizer
from src.core.types import Array


class GradientDescentOptimizer(Optimizer):
    """Vanilla Gradient Descent (SGD without momentum).

    Update rule:
        θ ← θ − η · ∇θ

    Args:
        learning_rate: Step size η.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def update(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """θ ← θ − η · ∇θ."""
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        """No internal state to reset."""
        raise NotImplementedError("TODO")
