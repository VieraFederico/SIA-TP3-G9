from src.metric.metric import Metric
from src.activation.activation import Array


class F1Metric(Metric):
    """F1 = 2 · precision · recall / (precision + recall)"""

    def compute(self, zeta: Array, O: Array) -> float:
        """F1 = 2 · P · R / (P + R)"""
        raise NotImplementedError("TODO")

    def name(self) -> str:
        return "f1"
