from src.core.optimizer import Optimizer
from src.core.types import Array


class AdamOptimizer(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation).

    Update rule (Kingma & Ba, 2015):
        m ← β1·m + (1−β1)·∇θ          # 1st moment (mean)
        v ← β2·v + (1−β2)·∇θ²          # 2nd moment (uncentered variance)
        m̂ = m / (1 − β1^t)             # bias-corrected
        v̂ = v / (1 − β2^t)
        θ ← θ − η · m̂ / (√v̂ + ε)

    Args:
        learning_rate: Step size η.  Defaults to 0.001.
        beta1: Exponential decay for the 1st moment.  Defaults to 0.9.
        beta2: Exponential decay for the 2nd moment.  Defaults to 0.999.
        epsilon: Numerical stability constant ε.  Defaults to 1e-8.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: list[Array] = []
        self._v: list[Array] = []
        self._t: int = 0

    def update(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """Apply Adam update and return new parameters."""
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        """Clear moment estimates and step counter."""
        raise NotImplementedError("TODO")
