from src.core.activation import ActivationFunction
from src.core.types import Array


class LogisticActivation(ActivationFunction):
    """Logistic (sigmoid) activation.

    θ(h)  = 1 / (1 + exp(−β · h))
    θ'(h) = β · θ(h) · (1 − θ(h))

    Args:
        beta: Slope parameter β > 0.  Defaults to 1.0.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def forward(self, h: Array) -> Array:
        """θ(h) = 1 / (1 + exp(−β · h))."""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """θ'(h) = β · θ(h) · (1 − θ(h))."""
        raise NotImplementedError("TODO")
