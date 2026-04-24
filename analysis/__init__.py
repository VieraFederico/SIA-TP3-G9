"""Offline analysis: load run artefacts and plot — no dependency on training or models."""

from analysis.loaders import (
    RunHistory,
    compare_runs,
    load_history,
    load_metrics,
    load_run,
)
from analysis.plots import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_curve,
    plot_metric_comparison,
)

__all__ = [
    "RunHistory",
    "compare_runs",
    "load_history",
    "load_metrics",
    "load_run",
    "plot_confusion_matrix",
    "plot_decision_boundary",
    "plot_loss_curve",
    "plot_metric_comparison",
]
