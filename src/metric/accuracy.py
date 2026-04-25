from src.metric.metric import Metric
from src.activation.activation import Array


class AccuracyMetric(Metric):
    """Fracción de ejemplos clasificados correctamente."""

    def compute(self, zeta: Array, O: Array) -> float:
        """accuracy = (1/N) Σ 1[argmax(ζᵢ) == argmax(Oᵢ)]"""
        raise NotImplementedError("TODO")

    def name(self) -> str:
        return "accuracy"
