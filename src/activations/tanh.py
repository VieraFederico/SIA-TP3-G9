from src.core.activation import ActivationFunction
from src.core.types import Array


class TanhActivation(ActivationFunction):
    """Scaled hyperbolic tangent activation.

    θ(h)  = tanh(β · h)
    θ'(h) = β · (1 − θ(h)²)

    Args:
        beta: Slope parameter β > 0.  Defaults to 1.0.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def forward(self, h: Array) -> Array:
        """θ(h) = tanh(β · h)."""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """θ'(h) = β · (1 − tanh²(β · h)) = β · (1 − θ(h)²)."""
        raise NotImplementedError("TODO")
