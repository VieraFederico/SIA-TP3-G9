from src.cost.cost import CostFunction
from src.activation.activation import Array


class MSECost(CostFunction):
    """E(ζ, O) = (1/2N) Σ (ζᵢ - Oᵢ)²"""

    def compute(self, zeta: Array, O: Array) -> float:
        """E = (1/2N) Σ (ζ - O)²"""
        raise NotImplementedError("TODO")

    def gradient(self, zeta: Array, O: Array) -> Array:
        """∂E/∂O = -(ζ - O) / N"""
        raise NotImplementedError("TODO")
