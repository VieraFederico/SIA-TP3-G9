import pandas as pd


def load_history(run_id: str) -> dict:
    """Lee results/<run_id>/history.json"""
    raise NotImplementedError("TODO")


def load_metrics(run_id: str) -> dict:
    """Lee results/<run_id>/metrics.json"""
    raise NotImplementedError("TODO")


def compare_runs(run_ids: list[str]) -> "pd.DataFrame":
    """Carga métricas de múltiples runs y las compara en un DataFrame."""
    raise NotImplementedError("TODO")
