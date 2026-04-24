from src.core.types import Array

from .metric import Metric


class AccuracyMetric(Metric):
    """Fraction of correctly classified samples."""

    def name(self) -> str:
        return "accuracy"

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """correct / total.  Applies argmax to multi-class predictions."""
        raise NotImplementedError("TODO")


class PrecisionMetric(Metric):
    """Precision = TP / (TP + FP).

    Args:
        average: ``"binary"``, ``"macro"``, or ``"weighted"``.
    """

    def __init__(self, average: str = "binary") -> None:
        self.average = average

    def name(self) -> str:
        return f"precision_{self.average}"

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """TP / (TP + FP), averaged according to ``self.average``."""
        raise NotImplementedError("TODO")


class RecallMetric(Metric):
    """Recall = TP / (TP + FN).

    Args:
        average: ``"binary"``, ``"macro"``, or ``"weighted"``.
    """

    def __init__(self, average: str = "binary") -> None:
        self.average = average

    def name(self) -> str:
        return f"recall_{self.average}"

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """TP / (TP + FN), averaged according to ``self.average``."""
        raise NotImplementedError("TODO")


class F1Metric(Metric):
    """F1 Score = 2 · Precision · Recall / (Precision + Recall).

    Args:
        average: ``"binary"``, ``"macro"``, or ``"weighted"``.
    """

    def __init__(self, average: str = "binary") -> None:
        self.average = average

    def name(self) -> str:
        return f"f1_{self.average}"

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """2·P·R / (P+R), averaged according to ``self.average``."""
        raise NotImplementedError("TODO")
