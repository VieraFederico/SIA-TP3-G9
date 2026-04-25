from src.metric.metric import Metric
from src.activation.activation import Array


class MSEMetric(Metric):
    """MSE como métrica de evaluación: (1/N) Σ (ζᵢ - Oᵢ)²"""

    def compute(self, zeta: Array, O: Array) -> float:
        """mse = (1/N) Σ (ζ - O)²"""
        raise NotImplementedError("TODO")

    def name(self) -> str:
        return "mse"
