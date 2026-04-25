from src.cost.cost import CostFunction
from src.activation.activation import Array


class CategoricalCrossEntropyCost(CostFunction):
    """E(ζ, O) = -1/N Σᵢ Σₖ ζᵢₖ log(Oᵢₖ)"""

    def compute(self, zeta: Array, O: Array) -> float:
        """E = -1/N Σ Σ ζ log(O)"""
        raise NotImplementedError("TODO")

    def gradient(self, zeta: Array, O: Array) -> Array:
        """∂E/∂O = -ζ / O / N"""
        raise NotImplementedError("TODO")
