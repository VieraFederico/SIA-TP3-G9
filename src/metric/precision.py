from src.metric.metric import Metric
from src.activation.activation import Array


class PrecisionMetric(Metric):
    """Precisión = TP / (TP + FP). Para clasificación binaria o macro-averaged."""

    def compute(self, zeta: Array, O: Array) -> float:
        """precision = TP / (TP + FP)"""
        raise NotImplementedError("TODO")

    def name(self) -> str:
        return "precision"
