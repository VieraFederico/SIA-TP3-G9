from src.core.optimizer import Optimizer
from src.core.types import Array


class MomentumOptimizer(Optimizer):
    """Gradient Descent with classical Momentum.

    Update rule:
        v ← β · v − η · ∇θ
        θ ← θ + v

    Args:
        learning_rate: Step size η.
        beta: Momentum coefficient β ∈ [0, 1).  Defaults to 0.9.
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.beta = beta
        self._velocities: list[Array] = []

    def update(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """v ← β·v − η·∇θ ; θ ← θ + v."""
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        """Clear velocity buffers."""
        raise NotImplementedError("TODO")
