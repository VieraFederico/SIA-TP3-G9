from src.activation.activation import ActivationFunction, Array


class ReLUActivation(ActivationFunction):
    """θ(h) = max(0, h),  θ'(h) = 1 if h > 0 else 0"""

    def compute(self, h: Array) -> Array:
        """θ(h) = max(0, h)"""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """θ'(h) = 1 if h > 0 else 0"""
        raise NotImplementedError("TODO")
