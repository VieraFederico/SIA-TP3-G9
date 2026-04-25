from src.activation.activation import ActivationFunction, Array


class LogisticActivation(ActivationFunction):
    """θ(h) = 1 / (1 + e^(-β·h)),  θ'(h) = β·θ(h)·(1 - θ(h))"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, h: Array) -> Array:
        """θ(h) = 1 / (1 + e^(-β·h))"""
        raise NotImplementedError("TODO")

    def derivative(self, h: Array) -> Array:
        """θ'(h) = β·θ(h)·(1 - θ(h))"""
        raise NotImplementedError("TODO")
