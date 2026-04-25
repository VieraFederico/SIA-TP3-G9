from src.activation.activation import ActivationFunction, Array


class StepActivation(ActivationFunction):
    """θ(h) = 1 if h >= 0 else 0. No diferenciable."""

    def compute(self, h: Array) -> Array:
        """θ(h) = 1 if h >= 0 else 0"""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """No definida para step. Retorna ceros por convención."""
        raise NotImplementedError("TODO")

    def is_differentiable(self) -> bool:
        return False
