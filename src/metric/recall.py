from src.metric.metric import Metric
from src.activation.activation import Array


class RecallMetric(Metric):
    """Recall = TP / (TP + FN). Para clasificación binaria o macro-averaged."""

    def compute(self, zeta: Array, O: Array) -> float:
        """recall = TP / (TP + FN)"""
        raise NotImplementedError("TODO")

    def name(self) -> str:
        return "recall"
