from src.core.types import Array

from .metric import Metric


class MSEMetric(Metric):
    """Mean Squared Error: (1/N) · Σ (y_pred − y_true)²."""

    def name(self) -> str:
        return "mse"

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """(1/N) · Σ (y_pred − y_true)²."""
        raise NotImplementedError("TODO")


class MAEMetric(Metric):
    """Mean Absolute Error: (1/N) · Σ |y_pred − y_true|."""

    def name(self) -> str:
        return "mae"

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """(1/N) · Σ |y_pred − y_true|."""
        raise NotImplementedError("TODO")


class R2Metric(Metric):
    """Coefficient of Determination R²: 1 − SS_res / SS_tot."""

    def name(self) -> str:
        return "r2"

    def compute(self, y_true: Array, y_pred: Array) -> float:
        """R² = 1 − Σ(y_true − y_pred)² / Σ(y_true − ȳ)²."""
        raise NotImplementedError("TODO")
