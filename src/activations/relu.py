from src.core.activation import ActivationFunction
from src.core.types import Array


class ReLUActivation(ActivationFunction):
    """Rectified Linear Unit activation.

    θ(h)  = max(0, h)
    θ'(h) = 1 if h > 0 else 0
    """

    def forward(self, h: Array) -> Array:
        """θ(h) = max(0, h)."""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """θ'(h) = 1 if h > 0 else 0."""
        raise NotImplementedError("TODO")
