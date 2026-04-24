from src.core.activation import ActivationFunction
from src.core.types import Array


class IdentityActivation(ActivationFunction):
    """Linear / identity activation used for ADALINE and regression outputs.

    θ(h) = h
    θ'(h) = 1
    """

    def forward(self, h: Array) -> Array:
        """θ(h) = h."""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """θ'(h) = 1 (array of ones, same shape as h)."""
        raise NotImplementedError("TODO")
