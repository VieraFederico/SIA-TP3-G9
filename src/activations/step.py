from src.core.activation import ActivationFunction
from src.core.types import Array


class StepActivation(ActivationFunction):
    """Heaviside step function used by the classic Simple Perceptron.

    θ(h) = 1 if h >= 0 else 0

    Not differentiable, so ``is_differentiable`` returns ``False`` and the
    derivative is undefined (raises ``NotImplementedError``).
    """

    def forward(self, h: Array) -> Array:
        """θ(h) = 1 if h >= 0 else 0."""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """Undefined — StepActivation is not differentiable."""
        raise NotImplementedError("StepActivation has no meaningful derivative.")

    def is_differentiable(self) -> bool:
        return False
