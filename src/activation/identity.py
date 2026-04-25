from src.activation.activation import ActivationFunction, Array


class IdentityActivation(ActivationFunction):
    """θ(h) = h,  θ'(h) = 1"""

    def compute(self, h: Array) -> Array:
        """θ(h) = h"""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """θ'(h) = 1"""
        raise NotImplementedError("TODO")
